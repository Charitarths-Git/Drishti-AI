"""
Step 2 — Scene Summary Generation via Qwen2.5-VL-7B-Instruct (Distributed)
============================================================================
Hardware target : 8× NVIDIA RTX A6000 (49 GB each)
Model           : Qwen/Qwen2.5-VL-7B-Instruct  (~16 GB VRAM per GPU, leaving 33 GB headroom)
Optimizations   : flash_attention_2, bfloat16, torch.compile, batch-per-GPU

Input  : frame_manifest.json  (from Step 1)
Output : scene_summaries.json  { video_id -> structured scene JSON }

Key design decisions:
  - One model instance per GPU (8 parallel workers via torch.multiprocessing)
  - bfloat16 — native Ampere precision, lower memory + faster than float16
  - flash_attention_2 — reduces attention memory O(n²) → O(n), ~3× faster on A6000
  - Checkpoint every 50 videos per GPU → safe to kill and resume
  - AVCaps-aware prompt: enriches audio_awareness field when audio captions exist
  - Rich spatial prompt: forces model to reason about navigable space + obstacle distance

Usage:
  # Single GPU (test)
  python step2_scene_summary.py --manifest frame_manifest.json --output scene_summaries.json

  # Distributed across all 8 A6000s
  python step2_scene_summary.py --manifest frame_manifest.json --output scene_summaries.json \\
      --distributed --n_gpus 8

  # Resume after interruption (automatic)
  python step2_scene_summary.py --manifest frame_manifest.json --output scene_summaries.json \\
      --distributed --n_gpus 8
"""

import os
import sys
import json
import time
import re
import base64
import argparse
import traceback
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# SCENE SUMMARY PROMPT
# Best practices applied:
#   1. Chain-of-thought trigger ("Think step by step…") improves spatial accuracy
#   2. Explicit distance estimates force metric reasoning
#   3. BLV-specific framing aligns the model's attention to navigation-critical detail
#   4. Structured JSON output with all fields prevents missing keys in downstream steps
# ─────────────────────────────────────────────────────────────────────────────

SCENE_PROMPT = """You are an expert assistant helping blind and low-vision (BLV) users navigate real environments safely.

Analyze the provided video frames carefully. Think step by step about what a sighted person sees and how to translate that into navigation-safe guidance for someone who cannot see.

Produce a detailed structured scene summary. Be specific about:
- EXACT object positions using clock-face directions and estimated metric distances
  (e.g. "chair at 12 o'clock, 1.5 meters ahead" not just "chair in front")
- ALL potential walking hazards, including low objects, steps, wet surfaces, cables, and narrow passages
- Safe corridors: where someone could walk 2+ meters without obstacles
- Human presence: number of people, direction of movement, proximity
- Lighting and visibility conditions that affect navigation

Return ONLY valid JSON — no markdown fences, no explanation text:
{
  "objects": ["exhaustive list of every visible object"],
  "spatial": {
    "object_name": "clock-face direction + estimated distance (e.g. 2 o'clock, 2m)"
  },
  "activity": "detailed description of all human activities and movement directions",
  "audio": "likely audio cues: speech, footsteps, machinery, traffic, alarms, nature sounds",
  "obstacles": [
    "precise description of each hazard + position (e.g. chair leg at 11 o'clock, 0.8m)"
  ],
  "safe_directions": [
    "specific safe walking directions with estimated clear distance (e.g. left at 9 o'clock, clear for 3m)"
  ],
  "scene_type": "one of: indoor_home, indoor_office, indoor_restaurant, indoor_shop, indoor_corridor, outdoor_street, outdoor_park, transit_station, vehicle, other",
  "lighting": "one of: bright_natural, bright_artificial, dim, dark, mixed",
  "floor_surface": "carpet, hardwood, tile, concrete, grass, gravel, wet, uneven, or combination",
  "crowd_density": "empty, sparse (1-3 people), moderate (4-10), crowded (10+)",
  "summary": "2-3 sentence plain-English description a blind user would hear via audio: scene type, immediate safety concerns, and one recommended action"
}"""


def _pick_caption(entry: dict, keys: list[str]) -> str:
    """Return first non-empty caption from a list of candidate field names.
    Handles both singular strings and lists (takes the first element)."""
    for k in keys:
        val = entry.get(k)
        if val:
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def build_avcaps_enhanced_prompt(scene_prompt: str, audio_caption: str = None, visual_caption: str = None) -> str:
    """
    If AVCaps captions are available, inject them into the prompt.
    This dramatically improves audio_awareness and activity fields
    since AVCaps captions are expert-written ground truth.
    """
    if not audio_caption and not visual_caption:
        return scene_prompt

    context_lines = [scene_prompt, "\nAdditional context from dataset annotations (use to enrich your response):"]
    if visual_caption:
        context_lines.append(f"Visual annotation: {visual_caption}")
    if audio_caption:
        context_lines.append(f"Audio annotation: {audio_caption}")
    context_lines.append("\nNow analyze the frames with this additional context in mind.")
    return "\n".join(context_lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING — robust extraction from messy LLM outputs
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """
    Robustly extract JSON from model output.
    Handles: clean JSON, markdown code fences, JSON embedded in prose,
    and TRUNCATED JSON (when max_new_tokens cuts off mid-array).
    """
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        return {"raw_response": text, "parse_error": True}

    # Try to find complete JSON object
    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i + 1])
                except json.JSONDecodeError:
                    break

    # --- Truncation recovery ---
    # JSON was cut off mid-generation. Extract whatever fields parsed correctly
    # by closing open brackets/braces and retrying.
    partial = text[brace_start:]

    # Close any open arrays and objects
    open_brackets = partial.count("[") - partial.count("]")
    open_braces   = partial.count("{") - partial.count("}")

    # Strip trailing incomplete token (e.g. last item cut mid-string)
    # Find last complete value boundary: comma, closing bracket/brace, or quote
    last_clean = max(
        partial.rfind(","),
        partial.rfind("}"),
        partial.rfind("]"),
    )
    if last_clean > 0:
        partial = partial[:last_clean]

    # Re-count after trimming
    open_brackets = partial.count("[") - partial.count("]")
    open_braces   = partial.count("{") - partial.count("}")

    partial += "]" * max(0, open_brackets)
    partial += "}" * max(0, open_braces)

    try:
        recovered = json.loads(partial)
        recovered["_truncated"] = True   # flag so we know this was partial
        return recovered
    except json.JSONDecodeError:
        pass

    # Last resort: extract key-value pairs with regex
    result = {"_truncated": True, "parse_error": True}
    for key in ["objects", "spatial", "activity", "audio", "obstacles",
                "safe_directions", "scene_type", "lighting", "floor_surface",
                "crowd_density", "summary"]:
        m = re.search(rf'"{key}"\s*:\s*(".*?"|{{.*?}}|\[.*?\])', text, re.DOTALL)
        if m:
            try:
                result[key] = json.loads(m.group(1))
            except Exception:
                result[key] = m.group(1).strip('"')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# QWEN2.5-VL INFERENCE — single GPU worker
# ─────────────────────────────────────────────────────────────────────────────

def load_qwen_model(gpu_id: int, model_name: str):
    """Load Qwen2.5-VL on a specific GPU with all A6000 optimizations."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading {model_name}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        local_files_only=True,
    )
    model.eval()

    # torch.compile for additional ~20% throughput on static inference shapes
    # Comment out if you hit compilation errors on first run
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass  # compile is optional

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        local_files_only=True,
    )

    print(f"[GPU {gpu_id}] Model loaded. VRAM: {torch.cuda.memory_allocated(gpu_id) / 1e9:.1f} GB")
    return model, processor


def run_qwen_inference(
    model,
    processor,
    frames: list[str],
    prompt_text: str,
    max_frames: int = 12,
    gpu_id: int = 0,
) -> str:
    """
    Run single-video inference on Qwen2.5-VL.

    Frame selection: up to max_frames evenly sampled from available frames.
    12 frames per video is the sweet spot for Qwen2.5-VL-7B:
      - More context than 8 frames for long videos
      - Within the model's visual context budget
    """
    import torch
    from qwen_vl_utils import process_vision_info

    # Sub-sample frames for this call
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]

    # Build message content: text prompt + all frame images
    content = [{"type": "text", "text": prompt_text}]
    for frame_path in frames:
        if Path(frame_path).exists():
            content.append({
                "type": "image",
                "image": f"file://{os.path.abspath(frame_path)}",
            })

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(f"cuda:{gpu_id}")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=600,    # scene JSON typically 400-550 tokens, 600 is safe ceiling
            do_sample=False,       # greedy decoding — fastest, no temperature needed
            repetition_penalty=1.1,
        )

    # Trim prompt tokens from output
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED WORKER (one per GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(
    gpu_id: int,
    video_ids: list[str],
    manifest: dict,
    combined_index: dict,
    output_path: str,
    model_name: str,
    max_frames: int,
):
    """
    Worker function for one GPU.
    Processes its assigned shard of video_ids, writing a per-GPU checkpoint file.
    The main process merges all checkpoint files.
    """
    checkpoint_path = output_path.replace(".json", f"_gpu{gpu_id}.json")

    # Resume from checkpoint
    done = {}
    if Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path) as f:
                done = json.load(f)
            print(f"[GPU {gpu_id}] Resuming: {len(done)} already done")
        except Exception:
            done = {}

    todo = [vid for vid in video_ids if vid not in done]
    print(f"[GPU {gpu_id}] Processing {len(todo)} videos")

    if not todo:
        return

    # Stagger model loads to avoid all 8 GPUs hammering shared memory simultaneously
    import time
    time.sleep(gpu_id * 20)

    try:
        model, processor = load_qwen_model(gpu_id, model_name)
    except Exception as e:
        import traceback
        print(f"[GPU {gpu_id}] FATAL: model load failed: {e}")
        traceback.print_exc()
        return

    for i, video_id in enumerate(tqdm(todo, desc=f"GPU {gpu_id}", position=gpu_id)):
        info = manifest.get(video_id, {})
        frames = info.get("frames", [])

        if not frames:
            done[video_id] = {"video_id": video_id, "error": "no_frames", "scene": {}}
            continue

        # Check for AVCaps annotations to enrich the prompt
        # combined_index keys are prefixed (avcaps__ID, charades__ID) but manifest keys are raw IDs
        index_entry = (
            combined_index.get(video_id)
            or combined_index.get(f"avcaps__{video_id}")
            or combined_index.get(f"charades__{video_id}")
            or {}
        )
        audio_caption = _pick_caption(index_entry, [
            "audio_caption", "audio_visual_caption",
            "audio_visual_captions", "GPT_AV_captions",
        ])
        visual_caption = _pick_caption(index_entry, [
            "visual_caption", "descriptions", "visual_captions",
        ])

        prompt = build_avcaps_enhanced_prompt(SCENE_PROMPT, audio_caption, visual_caption)

        try:
            raw = run_qwen_inference(
                model, processor, frames, prompt,
                max_frames=max_frames,
                gpu_id=gpu_id,
            )
            scene = parse_json_response(raw)

            done[video_id] = {
                "video_id": video_id,
                "scene": scene,
                "frames_used": frames[:max_frames],
                "dataset": index_entry.get("dataset", "unknown"),
                "scene_type_raw": index_entry.get("scene", ""),
                "had_avcaps_audio": bool(audio_caption),
            }

        except Exception as e:
            print(f"[GPU {gpu_id}] Error on {video_id}: {e}")
            done[video_id] = {"video_id": video_id, "error": str(e), "scene": {}}

        # Checkpoint every 50 videos
        if (i + 1) % 50 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(done, f, indent=2)

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(done, f, indent=2)
    print(f"[GPU {gpu_id}] Done. Results in {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_scene_summaries(
    manifest_path: str,
    output_path: str = "scene_summaries.json",
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_frames_per_video: int = 12,
    gpu_id: int = 0,
    resume: bool = True,
    combined_index_path: str = None,
):
    """Single-GPU processing (for testing or when running manually per GPU)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    combined_index = {}
    if combined_index_path and Path(combined_index_path).exists():
        with open(combined_index_path) as f:
            combined_index = json.load(f)
        print(f"Loaded combined index: {len(combined_index)} entries")

    existing = {}
    if resume and Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already processed")

    to_process = {vid: info for vid, info in manifest.items() if vid not in existing}
    print(f"Processing {len(to_process)} videos on GPU {gpu_id}")

    try:
        model, processor = load_qwen_model(gpu_id, model_name)
    except Exception as e:
        print(f"Model load failed: {e}")
        print("Make sure: pip install torch transformers accelerate qwen-vl-utils flash-attn")
        return

    summaries = dict(existing)
    errors = {}

    for video_id, info in tqdm(to_process.items(), desc="Generating summaries"):
        frames = info.get("frames", [])
        if not frames:
            errors[video_id] = "no_frames"
            continue

        index_entry = (
            combined_index.get(video_id)
            or combined_index.get(f"avcaps__{video_id}")
            or combined_index.get(f"charades__{video_id}")
            or {}
        )
        audio_caption = _pick_caption(index_entry, [
            "audio_caption", "audio_visual_caption",
            "audio_visual_captions", "GPT_AV_captions",
        ])
        visual_caption = _pick_caption(index_entry, [
            "visual_caption", "descriptions", "visual_captions",
        ])
        prompt = build_avcaps_enhanced_prompt(SCENE_PROMPT, audio_caption, visual_caption)

        try:
            raw = run_qwen_inference(
                model, processor, frames, prompt,
                max_frames=max_frames_per_video,
                gpu_id=gpu_id,
            )
            scene = parse_json_response(raw)

            summaries[video_id] = {
                "video_id": video_id,
                "scene": scene,
                "frames_used": frames[:max_frames_per_video],
                "dataset": index_entry.get("dataset", "unknown"),
                "had_avcaps_audio": bool(audio_caption),
            }

        except Exception as e:
            print(f"  [ERROR] {video_id}: {e}")
            errors[video_id] = str(e)

        if len(summaries) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(summaries, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n  Summaries : {len(summaries):,}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Output    : {output_path}")

    if errors:
        err_path = output_path.replace(".json", "_errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)

    return summaries


def run_distributed(
    manifest_path: str,
    output_path: str,
    model_name: str,
    n_gpus: int,
    max_frames: int,
    combined_index_path: str = None,
):
    """
    Distribute work across n_gpus using torch.multiprocessing.
    Each GPU gets a non-overlapping shard of video_ids.
    Checkpoints are per-GPU; this function merges them at the end.
    """
    import torch.multiprocessing as mp

    with open(manifest_path) as f:
        manifest = json.load(f)

    combined_index = {}
    if combined_index_path and Path(combined_index_path).exists():
        with open(combined_index_path) as f:
            combined_index = json.load(f)

    # Check which are already done in merged output
    existing = {}
    if Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Already merged: {len(existing)} videos")

    todo = [vid for vid in manifest if vid not in existing]
    print(f"Distributing {len(todo)} videos across {n_gpus} GPUs")

    # Shard: round-robin to balance load
    shards = [[] for _ in range(n_gpus)]
    for i, vid in enumerate(todo):
        shards[i % n_gpus].append(vid)

    for i, shard in enumerate(shards):
        print(f"  GPU {i}: {len(shard)} videos")

    # Launch workers
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(n_gpus):
        if not shards[gpu_id]:
            continue
        p = ctx.Process(
            target=_worker_fn,
            args=(
                gpu_id,
                shards[gpu_id],
                manifest,
                combined_index,
                output_path,
                model_name,
                max_frames,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge all GPU checkpoints
    print("\nMerging GPU checkpoint files...")
    merged = dict(existing)
    for gpu_id in range(n_gpus):
        cp = output_path.replace(".json", f"_gpu{gpu_id}.json")
        if Path(cp).exists():
            with open(cp) as f:
                shard_data = json.load(f)
            merged.update(shard_data)
            print(f"  GPU {gpu_id}: {len(shard_data)} entries")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    errors = [v for v in merged.values() if v.get("error")]
    avcaps_enriched = sum(1 for v in merged.values() if v.get("had_avcaps_audio"))

    print(f"\n  Total summaries   : {len(merged):,}")
    print(f"  AVCaps-enriched   : {avcaps_enriched:,}")
    print(f"  Errors            : {len(errors)}")
    print(f"  Output            : {output_path}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Generate scene summaries with Qwen2.5-VL (distributed A6000)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="frame_manifest.json from Step 1")
    parser.add_argument("--output", default="scene_summaries.json")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen VL model name. 7B fits one A6000; 72B needs all 8.",
    )
    parser.add_argument("--max_frames", type=int, default=12,
                        help="Frames per video sent to VLM (12 = optimal for 7B)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU to use for single-GPU mode")
    parser.add_argument("--distributed", action="store_true",
                        help="Use all --n_gpus in parallel (recommended)")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--combined_index",
        default=None,
        help="combined_index.json from download_datasets.py (enables AVCaps prompt enrichment)",
    )
    args = parser.parse_args()

    if args.distributed:
        run_distributed(
            manifest_path=args.manifest,
            output_path=args.output,
            model_name=args.model,
            n_gpus=args.n_gpus,
            max_frames=args.max_frames,
            combined_index_path=args.combined_index,
        )
    else:
        generate_scene_summaries(
            manifest_path=args.manifest,
            output_path=args.output,
            model_name=args.model,
            max_frames_per_video=args.max_frames,
            gpu_id=args.gpu_id,
            resume=not args.no_resume,
            combined_index_path=args.combined_index,
        )
