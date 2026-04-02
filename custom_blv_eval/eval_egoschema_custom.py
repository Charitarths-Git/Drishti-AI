"""
eval_egoschema_custom.py
========================
Evaluate ANY VLM on EgoSchema benchmark WITHOUT lmms-eval.
Bypasses lmms-eval's restrictive model registry so InternVL3 and Tarsier2 work!

Usage:
    python eval_egoschema_custom.py \
        --model_id internvl3 \
        --n_samples 400 \
        --output ./egoschema_results_internvl3.json

Supported models: qwen7b, qwen72b, internvl3, tarsier2
"""

import os
import json
import glob
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# ─── ENV ──────────────────────────────────────────────────────────────────────
os.environ["HF_HOME"] = "/usershome/cs671_user3/models"
EGOSCHEMA_CACHE = "/usershome/cs671_user3/models/datasets"
EGOSCHEMA_VIDEOS = "/usershome/cs671_user3/models/egoschema"  # extracted videos

# ─── MODELS ───────────────────────────────────────────────────────────────────
MODELS = {
    "qwen7b": {
        "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "arch": "qwen",
        "load_in_8bit": False,
    },
    "qwen72b": {
        "path": "Qwen/Qwen2.5-VL-72B-Instruct",
        "arch": "qwen",
        "load_in_8bit": True,
    },
    "tarsier2": {
        "path": "omni-research/Tarsier2-Recap-7b",
        "arch": "qwen",  # Tarsier2 is Qwen2-VL based
        "load_in_8bit": False,
    },
    "internvl3": {
        "path": "OpenGVLab/InternVL3-78B",
        "arch": "internvl",
        "load_in_8bit": True,
    },
}


def extract_frames_from_video(video_path: str, n_frames: int = 8) -> list:
    """Extract n_frames evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def build_mc_prompt(question: str, options: list) -> str:
    """Build a multiple-choice question prompt."""
    labels = ["A", "B", "C", "D", "E"]
    prompt = f"{question}\n\n"
    for i, opt in enumerate(options):
        if i < len(labels):
            prompt += f"{labels[i]}. {opt}\n"
    prompt += "\nAnswer with the letter only (A, B, C, D, or E)."
    return prompt


def parse_answer(response: str) -> str:
    """Extract the answer letter from model response."""
    response = response.strip()
    # Direct single letter
    if len(response) == 1 and response.upper() in "ABCDE":
        return response.upper()
    # Starts with letter
    if len(response) >= 1 and response[0].upper() in "ABCDE":
        if len(response) == 1 or not response[1].isalpha():
            return response[0].upper()
    # Search for "answer is X" pattern
    import re
    match = re.search(r'(?:answer|option|choice)\s*(?:is\s*)?([A-E])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Search for standalone letter
    match = re.search(r'\b([A-E])\b', response)
    if match:
        return match.group(1).upper()
    return response[0].upper() if response else "A"


def load_model(cfg):
    """Load model and return (model, processor/tokenizer, arch)."""
    print(f"Loading {cfg['path']}...")

    if cfg["arch"] == "qwen":
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if cfg["load_in_8bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        processor = AutoProcessor.from_pretrained(cfg["path"])
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kwargs).eval()
        return model, processor, "qwen"

    elif cfg["arch"] == "internvl":
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

        kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if cfg["load_in_8bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        model = AutoModel.from_pretrained(cfg["path"], **kwargs).eval()
        return model, tokenizer, "internvl"


def run_inference(model, processor, arch, frames, prompt):
    """Run inference on a single sample."""
    device = "cuda:0"

    if arch == "qwen":
        from qwen_vl_utils import process_vision_info

        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})

        msgs = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen_ids, skip_special_tokens=True).strip()

    elif arch == "internvl":
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pixel_values = torch.stack([transform(f) for f in frames]).to(torch.bfloat16).to(device)
        question = "<image>\n" * len(frames) + prompt
        gen_config = dict(max_new_tokens=50, do_sample=False)
        response, _ = model.chat(processor, pixel_values, question, gen_config)
        return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = MODELS[args.model_id]

    # ─── Load EgoSchema MC dataset ────────────────────────────────────────
    print("Loading EgoSchema MC dataset...")
    ds = load_dataset("lmms-lab/egoschema", "MC", cache_dir=EGOSCHEMA_CACHE, split="test")
    print(f"  Total samples: {len(ds)}")

    # Subsample
    indices = list(range(len(ds)))
    random.Random(42).shuffle(indices)
    indices = indices[:args.n_samples]

    # ─── Load model ───────────────────────────────────────────────────────
    model, processor, arch = load_model(cfg)
    print("Model loaded!")

    # ─── Resume logic ─────────────────────────────────────────────────────
    results = []
    done_ids = set()
    if Path(args.output).exists():
        try:
            results = json.load(open(args.output))
            done_ids = {r["video_uid"] for r in results}
            print(f"Resuming: {len(done_ids)} already done")
        except:
            pass

    # ─── Evaluate ─────────────────────────────────────────────────────────
    correct = sum(1 for r in results if r.get("correct", False))
    total_done = len(results)

    for idx in tqdm(indices, desc=f"EgoSchema [{args.model_id}]"):
        sample = ds[idx]
        video_uid = sample.get("video_uid", sample.get("q_uid", str(idx)))

        if video_uid in done_ids:
            continue

        # Find video file
        video_path = None
        for ext in [".mp4", ".mkv", ".webm"]:
            candidate = os.path.join(EGOSCHEMA_VIDEOS, f"{video_uid}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            # Try glob search
            matches = glob.glob(os.path.join(EGOSCHEMA_VIDEOS, f"*{video_uid}*"))
            if matches:
                video_path = matches[0]

        if video_path is None:
            print(f"  SKIP: Video not found for {video_uid}")
            results.append({
                "video_uid": video_uid,
                "predicted": "X",
                "ground_truth": "X",
                "correct": False,
                "error": "video_not_found",
            })
            continue

        # Extract frames
        frames = extract_frames_from_video(video_path, args.n_frames)
        if not frames:
            print(f"  SKIP: Could not extract frames from {video_uid}")
            results.append({
                "video_uid": video_uid,
                "predicted": "X",
                "ground_truth": "X",
                "correct": False,
                "error": "frame_extraction_failed",
            })
            continue

        # Build prompt
        question = sample.get("question", "")
        options = []
        for key in ["option 0", "option 1", "option 2", "option 3", "option 4"]:
            if key in sample:
                options.append(sample[key])

        prompt = build_mc_prompt(question, options)
        gt_idx = sample.get("answer", 0)
        gt_letter = ["A", "B", "C", "D", "E"][gt_idx] if isinstance(gt_idx, int) else str(gt_idx)

        # Run model
        try:
            response = run_inference(model, processor, arch, frames, prompt)
            predicted = parse_answer(response)
            is_correct = predicted == gt_letter
        except Exception as e:
            print(f"  ERROR on {video_uid}: {e}")
            response = ""
            predicted = "X"
            is_correct = False

        if is_correct:
            correct += 1
        total_done += 1

        results.append({
            "video_uid": video_uid,
            "question": question,
            "predicted": predicted,
            "raw_response": response,
            "ground_truth": gt_letter,
            "correct": is_correct,
        })

        # Print running accuracy
        if total_done % 10 == 0:
            acc = correct / total_done * 100
            print(f"  Running accuracy: {correct}/{total_done} = {acc:.1f}%")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print final results
    total_evaluated = sum(1 for r in results if r.get("error") is None)
    total_correct = sum(1 for r in results if r.get("correct", False))
    accuracy = total_correct / total_evaluated * 100 if total_evaluated > 0 else 0

    print("\n" + "=" * 60)
    print(f"  EGOSCHEMA RESULTS: {cfg['path']}")
    print("=" * 60)
    print(f"  Samples evaluated: {total_evaluated}")
    print(f"  Correct:           {total_correct}")
    print(f"  Accuracy:          {accuracy:.2f}%")
    print(f"  Results saved to:  {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
