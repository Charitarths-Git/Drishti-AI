"""
inference_smolvla.py
====================
Distributed inference — runs fine-tuned SmolVLM on test samples across
multiple GPUs and saves predictions for MCF/NAF evaluation.

Frame selection modes:
  --frame_mode clip   : CLIP greedy farthest-point (server, best quality)
  --frame_mode luv    : LUV color-difference (mobile, ~0.05s, no ML model)
  --frame_mode uniform: Uniform sampling (fastest fallback)

Mobile deployment notes:
  - Use --frame_mode luv --image_size 224 --max_new_tokens 50
  - INT8 quantized model recommended (--load_in_8bit)
  - Drop CLIP entirely — saves 8-12s on mid-range phones

Usage:
    # Server (best quality, CLIP diversity)
    CUDA_VISIBLE_DEVICES=0,1 python inference_smolvla.py \
        --base_model /usershome/cs671_user3/models/SmolVLM-Instruct \
        --lora_path  /usershome/cs671_user3/smolvlm_blv_lora/checkpoint-500 \
        --data_path  /usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl \
        --output     ./inference_outputs.json \
        --n_samples  500 --distributed --n_gpus 2

    # Mobile-optimized (LUV diversity, 224x224, short output)
    python inference_smolvla.py \
        --base_model /usershome/cs671_user3/models/SmolVLM-Instruct \
        --lora_path  /usershome/cs671_user3/smolvlm_blv_lora/checkpoint-500 \
        --data_path  /usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl \
        --output     ./inference_mobile.json \
        --frame_mode luv --image_size 224 --max_new_tokens 50 \
        --n_samples  200

    # From live video file (mobile deployment)
    python inference_smolvla.py \
        --base_model  /path/to/merged_model \
        --video_path  /path/to/video.mp4 \
        --query       "Can I walk forward safely?" \
        --frame_mode  luv --image_size 224 --max_new_tokens 50
"""

import os
import re
import json
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration as AutoModelForVision2Seq
from peft import PeftModel
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a navigation assistant for blind and low-vision users. "
    "Describe surroundings using clock-face directions (e.g. 12 o'clock = straight ahead, "
    "3 o'clock = right, 9 o'clock = left) and metric distances. "
    "Always mention obstacles, their positions, and safe walking directions. "
    "Be concise and actionable."
)

# ─────────────────────────────────────────────────────────────────────────────
# Frame selection — 3 modes
# ─────────────────────────────────────────────────────────────────────────────

def select_frames_uniform(image_paths: list, n: int, image_size: int) -> list:
    """Uniform sampling — zero cost, good enough for evenly-paced videos."""
    if len(image_paths) <= n:
        paths = image_paths
    else:
        step  = len(image_paths) // n
        paths = [image_paths[i * step] for i in range(n)]
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB").resize((image_size, image_size), Image.LANCZOS))
        except Exception:
            images.append(Image.new("RGB", (image_size, image_size), (128, 128, 128)))
    return images


def select_frames_luv(image_paths: list, n: int, image_size: int, pool_size: int = 12) -> list:
    """
    LUV color-difference diversity selection.
    No ML model needed — pure math, ~0.05s on any phone.
    Picks frames at peaks of inter-frame color difference signal.
    Same approach as the paper (Section 3.3).
    """
    if len(image_paths) <= n:
        return select_frames_uniform(image_paths, n, image_size)

    # Sample a pool of candidate frames
    pool_size = min(pool_size, len(image_paths))
    step      = len(image_paths) // pool_size
    pool_paths = [image_paths[i * step] for i in range(pool_size)]

    # Load as BGR numpy arrays for cv2 LUV conversion
    frames_bgr = []
    valid_paths = []
    for p in pool_paths:
        try:
            img = cv2.imread(p)
            if img is not None:
                frames_bgr.append(img)
                valid_paths.append(p)
        except Exception:
            pass

    if len(frames_bgr) <= n:
        return select_frames_uniform(image_paths, n, image_size)

    # Convert to LUV and compute inter-frame absolute difference
    frames_luv = [cv2.cvtColor(f, cv2.COLOR_BGR2Luv).astype(np.float32) for f in frames_bgr]
    diffs = [0.0]
    for i in range(1, len(frames_luv)):
        diffs.append(float(np.mean(np.abs(frames_luv[i] - frames_luv[i - 1]))))

    # Hanning window smoothing (reduces noise at edges)
    diffs   = np.array(diffs)
    window  = np.hanning(len(diffs))
    smoothed = diffs * window

    # Always include first frame, then pick peaks
    selected = [0]
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((smoothed[i], i))

    peaks.sort(reverse=True)
    for _, idx in peaks[:n - 1]:
        if idx not in selected:
            selected.append(idx)

    # Fill remaining with uniform if not enough peaks
    if len(selected) < n:
        s = len(frames_bgr) // n
        for i in range(1, n):
            c = i * s
            if c not in selected and len(selected) < n:
                selected.append(c)

    selected = sorted(selected[:n])

    # Return as resized PIL images
    result = []
    for idx in selected:
        try:
            img = Image.fromarray(cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB))
            result.append(img.resize((image_size, image_size), Image.LANCZOS))
        except Exception:
            result.append(Image.new("RGB", (image_size, image_size), (128, 128, 128)))
    return result


def select_frames_clip(image_paths: list, n: int, image_size: int, device: str) -> list:
    """
    CLIP greedy farthest-point sampling — best diversity, server only.
    Requires ~150MB CLIP model. Takes 8-12s on mid-range mobile — do NOT use on mobile.
    """
    if len(image_paths) <= n:
        imgs = []
        for p in image_paths:
            try:
                imgs.append(Image.open(p).convert("RGB").resize((image_size, image_size), Image.LANCZOS))
            except Exception:
                imgs.append(Image.new("RGB", (image_size, image_size), (128, 128, 128)))
        return imgs

    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_proc  = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/usershome/cs671_user3/models/clip",
            local_files_only=True,
        )
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/usershome/cs671_user3/models/clip",
            local_files_only=True,
        ).to(device).eval()

        images, valid = [], []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception:
                pass

        if len(valid) <= n:
            del clip_model
            torch.cuda.empty_cache()
            return select_frames_uniform(valid, n, image_size)

        inputs = clip_proc(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1).cpu().float()

        selected = [0]
        while len(selected) < n:
            sel    = feats[selected]
            sims   = feats @ sel.T
            maxsim = sims.max(dim=1).values
            maxsim[selected] = 1.0
            selected.append(int(maxsim.argmin()))

        del clip_model
        torch.cuda.empty_cache()

        result = []
        for i in sorted(selected):
            try:
                result.append(valid[i] if isinstance(valid[i], Image.Image)
                               else Image.open(valid[i]).convert("RGB").resize((image_size, image_size), Image.LANCZOS))
            except Exception:
                result.append(Image.new("RGB", (image_size, image_size), (128, 128, 128)))
        return result

    except Exception:
        return select_frames_uniform(image_paths, n, image_size)


def extract_frames_from_video(video_path: str, n_frames: int, image_size: int,
                               frame_mode: str = "luv") -> list:
    """
    Extract diverse frames directly from a video file.
    Used for live mobile deployment (not pre-extracted frames).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pool  = min(24, total)
    step  = max(1, total // pool)

    frames_bgr  = []
    frame_paths = []  # dummy — reuse LUV logic via temp PIL list

    for i in range(pool):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        return [Image.new("RGB", (image_size, image_size), (128, 128, 128))] * n_frames

    if frame_mode == "uniform" or len(frames_bgr) <= n_frames:
        step2  = max(1, len(frames_bgr) // n_frames)
        chosen = [frames_bgr[i * step2] for i in range(min(n_frames, len(frames_bgr)))]
    else:
        # LUV diversity on raw BGR frames
        frames_luv = [cv2.cvtColor(f, cv2.COLOR_BGR2Luv).astype(np.float32) for f in frames_bgr]
        diffs      = [0.0] + [float(np.mean(np.abs(frames_luv[i] - frames_luv[i-1])))
                               for i in range(1, len(frames_luv))]
        smoothed   = np.array(diffs) * np.hanning(len(diffs))

        selected = [0]
        peaks    = [(smoothed[i], i) for i in range(1, len(smoothed)-1)
                    if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]]
        peaks.sort(reverse=True)
        for _, idx in peaks[:n_frames - 1]:
            if idx not in selected:
                selected.append(idx)

        while len(selected) < n_frames:
            s = len(frames_bgr) // n_frames
            for i in range(1, n_frames):
                c = i * s
                if c not in selected and len(selected) < n_frames:
                    selected.append(c)

        chosen = [frames_bgr[i] for i in sorted(selected[:n_frames])]

    return [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((image_size, image_size), Image.LANCZOS)
        for f in chosen
    ]


def get_frames(sample: dict, n_frames: int, image_size: int,
               frame_mode: str, device: str) -> list:
    """
    Unified frame getter — handles both pre-extracted paths and video files.
    """
    paths = sample.get("frames", [])

    if frame_mode == "clip":
        return select_frames_clip(paths, n_frames, image_size, device)
    elif frame_mode == "luv":
        return select_frames_luv(paths, n_frames, image_size)
    else:
        return select_frames_uniform(paths, n_frames, image_size)


# ─────────────────────────────────────────────────────────────────────────────
# Response cleaning — removes markdown artifacts for mobile TTS
# ─────────────────────────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    """
    Strip markdown formatting and truncate at natural sentence boundary.
    Critical for mobile TTS — BLV users hear everything including ** and \n.
    """
    # Remove markdown bold/headers
    text = re.sub(r'\*\*.*?\*\*:?\s*', '', text)
    text = re.sub(r'#{1,3}\s+', '', text)
    # Remove self-generated Q&A sections
    text = re.sub(r'\n+(Question|Q|Note|Additional|Conclusion).*', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Collapse whitespace
    text = re.sub(r'\n+', ' ', text).strip()
    text = re.sub(r'\s{2,}', ' ', text)
    # Keep only first 2 sentences for mobile
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(sentences[:2]).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base_model_path: str, lora_path: str, device: str, load_in_8bit: bool = False):
    processor = AutoProcessor.from_pretrained(
        base_model_path, local_files_only=True, trust_remote_code=True
    )

    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            quantization_config=bnb,
            device_map=device,
            local_files_only=True,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
            device_map="cpu",
        )

    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(model, lora_path, device_map="cpu")
        model = model.merge_and_unload()

    if not load_in_8bit:
        model = model.to(device)

    model.eval()
    torch.cuda.empty_cache()
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Sample loading
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(data_path: str, n_samples: int, val_fraction: float = 0.02, seed: int = 42):
    raw = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item  = json.loads(line)
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue
            raw.append({
                "video_id":    item.get("video_id", ""),
                "frames":      item.get("images", []),
                "instruction": convs[0].get("value", ""),
                "reference":   convs[1].get("value", ""),
                "category":    item.get("category", ""),
                "quality":     item.get("quality_score", 0),
            })

    rng   = random.Random(seed)
    rng.shuffle(raw)
    n_val = max(1, int(len(raw) * val_fraction))
    pool  = raw[:n_val]

    if n_samples > len(pool):
        pool = pool + raw[n_val: n_val + (n_samples - len(pool))]

    return pool[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Inference worker (one per GPU)
# ─────────────────────────────────────────────────────────────────────────────

def inference_worker(
    gpu_id: int,
    samples: list,
    base_model: str,
    lora_path: str,
    output_path: str,
    n_frames: int,
    max_new_tokens: int,
    frame_mode: str,
    image_size: int,
    load_in_8bit: bool,
    mobile_clean: bool,
):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model (frame_mode={frame_mode}, size={image_size}x{image_size})...")
    model, processor = load_model(base_model, lora_path, device, load_in_8bit)
    print(f"[GPU {gpu_id}] Ready. Processing {len(samples)} samples...")

    results  = []
    done_ids = set()
    if Path(output_path).exists():
        try:
            existing = json.load(open(output_path))
            results  = existing
            done_ids = {r["video_id"] + r["instruction"] for r in results}
            print(f"[GPU {gpu_id}] Resuming: {len(done_ids)} already done")
        except Exception:
            pass

    remaining = [s for s in samples if (s["video_id"] + s["instruction"]) not in done_ids]

    for sample in tqdm(remaining, desc=f"GPU {gpu_id}", position=gpu_id):
        images = get_frames(sample, n_frames, image_size, frame_mode, device)

        try:
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in images]
                    + [{"type": "text", "text": sample["instruction"]}],
                },
            ]
            prompt = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=prompt, images=images, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            n_in       = inputs["input_ids"].shape[1]
            prediction = processor.tokenizer.decode(
                out_ids[0][n_in:], skip_special_tokens=True
            ).strip()

            if mobile_clean:
                prediction = clean_response(prediction)

        except Exception as e:
            prediction = ""
            print(f"  [GPU {gpu_id}] WARN {sample['video_id']}: {e}")

        results.append({
            "video_id":    sample["video_id"],
            "category":    sample["category"],
            "instruction": sample["instruction"],
            "reference":   sample["reference"],
            "prediction":  prediction,
            "frame_mode":  frame_mode,
            "image_size":  image_size,
        })

        if len(results) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    empty = sum(1 for r in results if not r["prediction"])
    print(f"[GPU {gpu_id}] Done. {len(results)} results, {empty} empty → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single video inference (mobile deployment entry point)
# ─────────────────────────────────────────────────────────────────────────────

def infer_single_video(args):
    """Run inference on a single video file with a user query."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model, processor = load_model(args.base_model, args.lora_path, device, args.load_in_8bit)

    print(f"Extracting {args.n_frames} frames from {args.video_path}...")
    images = extract_frames_from_video(
        args.video_path, args.n_frames, args.image_size, args.frame_mode
    )
    print(f"Got {len(images)} frames at {args.image_size}x{args.image_size}")

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images]
            + [{"type": "text", "text": args.query}],
        },
    ]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

    print("Generating response...")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    n_in       = inputs["input_ids"].shape[1]
    prediction = processor.tokenizer.decode(out_ids[0][n_in:], skip_special_tokens=True).strip()

    if args.mobile_clean:
        prediction = clean_response(prediction)

    print("\n" + "="*60)
    print(f"Query     : {args.query}")
    print(f"Response  : {prediction}")
    print(f"Tokens in : {n_in}  |  Tokens out: {out_ids.shape[1] - n_in}")
    print("="*60)
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--base_model",     default="/usershome/cs671_user3/models/SmolVLM-Instruct")
    p.add_argument("--lora_path",      default="./smolvlm_blv_lora")
    p.add_argument("--load_in_8bit",   action="store_true", help="INT8 quantization for mobile")

    # Data — dataset mode
    p.add_argument("--data_path",      default="/usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl")
    p.add_argument("--output",         default="./inference_outputs.json")
    p.add_argument("--n_samples",      type=int, default=500)

    # Data — single video mode
    p.add_argument("--video_path",     default=None,  help="Path to video file (mobile mode)")
    p.add_argument("--query",          default="What obstacles are ahead of me?")

    # Frame selection
    p.add_argument("--n_frames",       type=int,   default=4)
    p.add_argument("--frame_mode",     default="clip",
                   choices=["clip", "luv", "uniform"],
                   help="clip=best quality(server), luv=mobile, uniform=fastest")
    p.add_argument("--image_size",     type=int,   default=336,
                   help="Resize frames to this square size. Use 224 for mobile.")

    # Generation
    p.add_argument("--max_new_tokens", type=int,   default=50,
                   help="50 for mobile (~35 words), 150 for full eval")
    p.add_argument("--mobile_clean",   action="store_true",
                   help="Strip markdown and truncate to 2 sentences for TTS")

    # Distributed
    p.add_argument("--distributed",    action="store_true")
    p.add_argument("--n_gpus",         type=int,   default=1)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Single video mode (mobile deployment)
    if args.video_path:
        infer_single_video(args)
        return

    # Dataset eval mode
    samples = load_samples(args.data_path, args.n_samples, seed=args.seed)
    print(f"Loaded {len(samples)} samples | frame_mode={args.frame_mode} | "
          f"size={args.image_size}x{args.image_size} | max_new_tokens={args.max_new_tokens}")

    if args.distributed and args.n_gpus > 1:
        shards       = [[] for _ in range(args.n_gpus)]
        output_paths = [args.output.replace(".json", f"_gpu{i}.json") for i in range(args.n_gpus)]

        for i, s in enumerate(samples):
            shards[i % args.n_gpus].append(s)

        processes = []
        for gpu_id in range(args.n_gpus):
            p = mp.Process(
                target=inference_worker,
                args=(
                    gpu_id,
                    shards[gpu_id],
                    args.base_model,
                    args.lora_path,
                    output_paths[gpu_id],
                    args.n_frames,
                    args.max_new_tokens,
                    args.frame_mode,
                    args.image_size,
                    args.load_in_8bit,
                    args.mobile_clean,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_results = []
        for op in output_paths:
            if Path(op).exists():
                all_results.extend(json.load(open(op)))

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nMerged {len(all_results)} results → {args.output}")

    else:
        inference_worker(
            gpu_id=0,
            samples=samples,
            base_model=args.base_model,
            lora_path=args.lora_path,
            output_path=args.output,
            n_frames=args.n_frames,
            max_new_tokens=args.max_new_tokens,
            frame_mode=args.frame_mode,
            image_size=args.image_size,
            load_in_8bit=args.load_in_8bit,
            mobile_clean=args.mobile_clean,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
