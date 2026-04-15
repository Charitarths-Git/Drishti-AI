"""
train_smolvla.py  —  Optimized for 8× NVIDIA RTX A6000 (Ampere, 49 GB each)
=============================================================================
Fine-tunes SmolVLM-Instruct on 210K BLV navigation samples with LoRA.

Optimizations applied:
  ✓ flash_attention_2          — ~3x attention speedup on Ampere
  ✓ bfloat16 everywhere        — native Ampere precision, ~10-15% faster than fp16
  ✓ tf32 matmul                — free speedup on Ampere tensor cores
  ✓ torchrun DDP               — multi-GPU, true data parallelism
  ✓ batch_size=2 + accum=8     — fits 4600-token sequences per GPU
  ✓ dataloader pin_memory      — faster CPU→GPU transfers
  ✓ CLIP local cache           — no HF network calls during training
  ✓ gradient_checkpointing     — saves ~30% VRAM at ~15% speed cost
  ✓ use_cache=False            — required with grad checkpointing
  ✓ use_reentrant=False        — avoids silent autograd errors
  ✓ resume_from_checkpoint     — full restart protection
  ✓ marker_ids fixed           — "Assistant:" token search (not apply_chat_template)
  ✓ resize to 336×336          — ~5x patch reduction per image
  ✓ no processor truncation    — image tokens never cut

Token budget:
  4 frames × ~650 tokens = ~2,600 image tokens  (336×336 resize)
  + system/instruction/response ~350 tokens
  ≈ ~2,950 tokens/sample  (was ~13,600 at full resolution)

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 --master_port=29506 train_smolvla.py \\
        --model_path /usershome/cs671_user3/models/SmolVLM-Instruct \\
        --data_path  /usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl \\
        --output_dir /usershome/cs671_user3/smolvlm_blv_lora \\
        --epochs 3 --n_frames 4 --batch_size 2 --grad_accum 8

    # Resume after interruption
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 --master_port=29506 train_smolvla.py \\
        --model_path /usershome/cs671_user3/models/SmolVLM-Instruct \\
        --data_path  /usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl \\
        --output_dir /usershome/cs671_user3/smolvlm_blv_lora \\
        --resume     /usershome/cs671_user3/smolvlm_blv_lora/checkpoint-500
"""

import os
import json
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────────────────────
# Ampere-specific global optimizations (tf32)
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a navigation assistant for blind and low-vision users. "
    "Describe surroundings using clock-face directions (12 o'clock = ahead, "
    "3 o'clock = right, 9 o'clock = left) and metric distances. "
    "Always mention obstacles with positions and safe walking directions. "
    "Be concise and actionable."
)

N_FRAMES           = 4
IMAGE_RESIZE       = (336, 336)   # reduces Idefics3 patches from ~81 to ~9-16 per image
MAX_SEQ_LEN        = 4096
MAX_RESPONSE_WORDS = 120          # ~150 tokens

CLIP_CACHE_DIR     = "/usershome/cs671_user3/models/clip"


# ─────────────────────────────────────────────────────────────────────────────
# Image resize helper
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(img: Image.Image, size: tuple = IMAGE_RESIZE) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP diversity frame selector
# ─────────────────────────────────────────────────────────────────────────────

_clip_model     = None
_clip_processor = None
_clip_device    = None


def _ensure_clip(device: str):
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None and _clip_device == device:
        return True
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=CLIP_CACHE_DIR,
            local_files_only=True,
        )
        _clip_model = (
            CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=CLIP_CACHE_DIR,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        _clip_device = device
        return True
    except Exception:
        return False


def select_frames_clip(paths: list, n: int, device: str) -> list:
    """Greedy farthest-point sampling in CLIP embedding space."""
    if len(paths) <= n:
        return paths

    if not _ensure_clip(device):
        step = len(paths) // n
        return [paths[i * step] for i in range(n)]

    try:
        images, valid = [], []
        for p in paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception:
                pass

        if len(valid) <= n:
            return valid

        inp = _clip_processor(images=images, return_tensors="pt", padding=True)
        inp = {k: v.to(_clip_device) for k, v in inp.items()}

        with torch.no_grad():
            feats = _clip_model.get_image_features(**inp)
            feats = F.normalize(feats, dim=-1).cpu().float()

        selected = [0]
        while len(selected) < n:
            sel_feats = feats[selected]
            sims      = feats @ sel_feats.T
            max_sim   = sims.max(dim=1).values
            max_sim[selected] = 1.0
            selected.append(int(max_sim.argmin()))

        return [valid[i] for i in sorted(selected)]

    except Exception:
        step = len(paths) // n
        return [paths[i * step] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BLVDataset(Dataset):
    def __init__(
        self,
        path: str,
        n_frames: int       = N_FRAMES,
        min_quality: int    = 2,
        split: str          = "train",
        val_fraction: float = 0.02,
        seed: int           = 42,
        clip_device: str    = "cpu",
    ):
        self.n_frames    = n_frames
        self.clip_device = clip_device

        raw = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item  = json.loads(line)
                convs = item.get("conversations", [])

                if item.get("quality_score", 0) < min_quality:
                    continue
                if len(convs) < 2:
                    continue

                instr = convs[0].get("value", "").strip()
                resp  = convs[1].get("value", "").strip()

                if not instr or not resp:
                    continue

                words = resp.split()
                if len(words) > MAX_RESPONSE_WORDS:
                    resp = " ".join(words[:MAX_RESPONSE_WORDS])

                frames = item.get("images", [])
                if not frames:
                    continue

                raw.append({
                    "frames":      frames,
                    "instruction": instr,
                    "response":    resp,
                })

        rng = random.Random(seed)
        rng.shuffle(raw)
        n_val     = max(1, int(len(raw) * val_fraction))
        self.data = raw[:n_val] if split == "val" else raw[n_val:]

        print(f"[Dataset] {split}: {len(self.data):,} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        paths  = select_frames_clip(item["frames"], self.n_frames, self.clip_device)
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", IMAGE_RESIZE, (128, 128, 128))
            images.append(resize_image(img))

        return {
            "images":      images,
            "instruction": item["instruction"],
            "response":    item["response"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class BLVCollator:
    def __init__(self, processor, max_length: int = MAX_SEQ_LEN):
        self.processor  = processor
        self.max_length = max_length

        # "Assistant:" is the actual prefix in SmolVLM chat format
        # Verified by inspecting apply_chat_template output
        self._marker_ids = processor.tokenizer(
            "Assistant:", add_special_tokens=False
        )["input_ids"]

    def __call__(self, batch):
        texts, images = [], []

        for item in batch:
            msgs = [
                {
                    "role":    "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in item["images"]]
                        + [{"type": "text", "text": item["instruction"]}]
                    ),
                },
                {
                    "role":    "assistant",
                    "content": [{"type": "text", "text": item["response"]}],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(item["images"])

        # No truncation — Idefics3 image tokens must not be cut
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Build labels — mask everything before "Assistant:" response
        labels     = inputs["input_ids"].clone()
        marker_ids = self._marker_ids
        marker_len = len(marker_ids)

        for i in range(labels.shape[0]):
            ids = labels[i].tolist()
            sep = -1
            for j in range(len(ids) - marker_len):
                if ids[j: j + marker_len] == marker_ids:
                    sep = j + marker_len
                    break
            if sep > 0:
                labels[i, :sep] = -100
            else:
                labels[i, :] = -100   # marker not found — skip sample

        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs


# ─────────────────────────────────────────────────────────────────────────────
# Model — SmolVLM-Instruct with flash_attention_2 + bfloat16 + LoRA
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str, lora_r: int, lora_alpha: int):
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )

    print(f"Loading model from {model_path}...")
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )

    # Freeze vision encoder
    frozen = 0
    for name, param in model.named_parameters():
        if "vision_model" in name or "vision_encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen {frozen / 1e6:.1f}M vision encoder parameters")

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",  default="/usershome/cs671_user3/models/SmolVLM-Instruct")
    p.add_argument("--data_path",   default="/usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl")
    p.add_argument("--output_dir",  default="/usershome/cs671_user3/smolvlm_blv_lora")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--n_frames",    type=int,   default=N_FRAMES)
    p.add_argument("--max_length",  type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--val_split",   type=float, default=0.02)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--min_quality", type=int,   default=2)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0
    clip_dev   = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if is_main:
        print("=" * 65)
        print("  SmolVLM-Instruct BLV Fine-Tuning")
        print(f"  Model      : {args.model_path}")
        print(f"  Data       : {args.data_path}")
        print(f"  Output     : {args.output_dir}")
        print(f"  Epochs     : {args.epochs}")
        print(f"  Frames     : {args.n_frames} @ {IMAGE_RESIZE[0]}×{IMAGE_RESIZE[1]} (CLIP diversity + resize)")
        print(f"  Max seq    : {args.max_length} tokens")
        print(f"  LoRA       : r={args.lora_r}  alpha={args.lora_alpha}")
        print(f"  Batch/GPU  : {args.batch_size}  accum={args.grad_accum}")
        n_gpus = int(os.environ.get("WORLD_SIZE", 1))
        eff    = args.batch_size * args.grad_accum * n_gpus
        print(f"  Eff. batch : {eff}  ({args.batch_size} × {args.grad_accum} × {n_gpus} GPUs)")
        print(f"  Precision  : bfloat16 + flash_attention_2")
        print("=" * 65)

    model, processor = load_model_and_processor(
        args.model_path, args.lora_r, args.lora_alpha
    )

    train_ds = BLVDataset(
        args.data_path,
        n_frames=args.n_frames,
        min_quality=args.min_quality,
        split="train",
        val_fraction=args.val_split,
        seed=args.seed,
        clip_device=clip_dev,
    )
    val_ds = BLVDataset(
        args.data_path,
        n_frames=args.n_frames,
        min_quality=args.min_quality,
        split="val",
        val_fraction=args.val_split,
        seed=args.seed,
        clip_device=clip_dev,
    )

    collator = BLVCollator(processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=3,
        logging_steps=25,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if is_main:
        print("\nSaving LoRA adapter + processor...")
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Done → {args.output_dir}")


if __name__ == "__main__":
    main()
