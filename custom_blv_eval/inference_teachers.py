"""
inference_teachers.py
=====================
Runs the VLM teacher models on the local BLV dataset (avcaps/charades) extracting Frames
using LUV/uniform strategies and outputting predictions for evaluate_blv.py.

Usage:
    python inference_teachers.py \
        --model_id qwen7b \
        --data_path /usershome/cs671_user3/blv_pipeline/training_data/dataset.jsonl \
        --output ./inference_outputs_qwen7b.json \
        --n_samples 400
"""

import os
import json
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel

# ─── MODELS CONFIG ────────────────────────────────────────────────────────────
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
        "arch": "qwen",
        "load_in_8bit": False,
    },
    "internvl3": {
        "path": "OpenGVLab/InternVL3-78B",
        "arch": "internvl",
        "load_in_8bit": True,
    }
}

SYSTEM_PROMPT = (
    "You are a navigation assistant for blind and low-vision users. "
    "Describe surroundings using clock-face directions (e.g. 12 o'clock = straight ahead, "
    "3 o'clock = right, 9 o'clock = left) and metric distances. "
    "Always mention obstacles, their positions, and safe walking directions. "
    "Be concise and actionable."
)

# ─── FRAME SELECTION (From inference_smolvla.py) ─────────────────────────────

def select_frames_uniform(image_paths: list, n: int) -> list:
    if len(image_paths) <= n:
        return image_paths
    step = len(image_paths) // n
    return [image_paths[i * step] for i in range(n)]

def select_frames_luv(image_paths: list, n: int, pool_size: int = 24) -> list:
    if len(image_paths) <= n:
        return select_frames_uniform(image_paths, n)
    
    pool_size = min(pool_size, len(image_paths))
    step = len(image_paths) // pool_size
    pool_paths = [image_paths[i * step] for i in range(pool_size)]
    
    frames_bgr, valid_paths = [], []
    for p in pool_paths:
        try:
            img = cv2.imread(p)
            if img is not None:
                frames_bgr.append(img)
                valid_paths.append(p)
        except Exception:
            pass
            
    if len(frames_bgr) <= n:
        return select_frames_uniform(valid_paths, n)
        
    frames_luv = [cv2.cvtColor(f, cv2.COLOR_BGR2Luv).astype(np.float32) for f in frames_bgr]
    diffs = [0.0]
    for i in range(1, len(frames_luv)):
        diffs.append(float(np.mean(np.abs(frames_luv[i] - frames_luv[i-1]))))
        
    smoothed = np.array(diffs) * np.hanning(len(diffs))
    
    selected = [0]
    peaks = [(smoothed[i], i) for i in range(1, len(smoothed)-1) 
             if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]]
    peaks.sort(reverse=True)
    
    for _, idx in peaks[:n-1]:
        if idx not in selected:
            selected.append(idx)
            
    if len(selected) < n:
        s = len(frames_bgr) // n
        for i in range(1, n):
            c = i * s
            if c not in selected and len(selected) < n:
                selected.append(c)
                
    return [valid_paths[i] for i in sorted(selected[:n])]

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--frame_mode", choices=["uniform", "luv"], default="luv")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    cfg = MODELS[args.model_id]
    device = "cuda:0"
    
    print(f"Loading {cfg['path']}...")
    
    if cfg["arch"] == "qwen":
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("ERROR: For Qwen models, run: pip install qwen-vl-utils")
            return
            
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if cfg["load_in_8bit"]:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
        processor = AutoProcessor.from_pretrained(cfg["path"])
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], **kwargs)
        
    elif cfg["arch"] == "internvl":
        import torchvision.transforms as T
        from transformers import BitsAndBytesConfig
        kwargs = {
            "device_map": "auto", 
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True
        }
        if cfg["load_in_8bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
        model = AutoModel.from_pretrained(cfg["path"], **kwargs).eval()
        
    print("Model loaded successfully.")

    # Load samples
    raw = []
    with open(args.data_path) as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            convs = item.get("conversations", [])
            if len(convs) < 2: continue
            raw.append({
                "video_id": item.get("video_id", ""),
                "frames": item.get("images", []),
                "instruction": convs[0].get("value", ""),
                "reference": convs[1].get("value", ""),
                "category": item.get("category", "")
            })
            
    rng = random.Random(42)
    rng.shuffle(raw)
    samples = raw[:args.n_samples]
    
    # Resume logic
    results, done_ids = [], set()
    if Path(args.output).exists():
        try:
            results = json.load(open(args.output))
            done_ids = {r["video_id"] + r["instruction"] for r in results}
            print(f"Resuming: {len(done_ids)} already done")
        except: pass

    remaining = [s for s in samples if (s["video_id"] + s["instruction"]) not in done_ids]
    
    for sample in tqdm(remaining, desc="Evaluating"):
        paths = sample.get("frames", [])
        if args.frame_mode == "luv":
            sel_paths = select_frames_luv(paths, args.n_frames)
        else:
            sel_paths = select_frames_uniform(paths, args.n_frames)
            
        prediction = ""
        try:
            if cfg["arch"] == "qwen":
                content = [{"type": "image", "image": f"file://{p}"} for p in sel_paths]
                content.append({"type": "text", "text": sample["instruction"]})
                
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ]
                
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msgs)
                inputs = processor(
                    text=[text], 
                    images=image_inputs, 
                    videos=video_inputs, 
                    padding=True, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                prediction = processor.decode(gen_ids, skip_special_tokens=True).strip()
                
            elif cfg["arch"] == "internvl":
                # InternVL logic: load images and pass to model.chat
                from PIL import Image
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                pixel_values = []
                for p in sel_paths:
                    img = Image.open(p).convert("RGB")
                    pixel_values.append(transform(img))
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(device)
                
                question = f"{SYSTEM_PROMPT}\n\n" + "<image>\n" * len(sel_paths) + sample["instruction"]
                gen_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
                
                response, _ = model.chat(tokenizer, pixel_values, question, gen_config)
                prediction = response.strip()
                
        except Exception as e:
            print(f"Error on {sample['video_id']}: {e}")
            
        results.append({
            "video_id": sample["video_id"],
            "category": sample["category"],
            "instruction": sample["instruction"],
            "reference": sample["reference"],
            "prediction": prediction
        })
        
        if len(results) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
                
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
