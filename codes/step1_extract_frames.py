"""
Step 1 — Frame Extraction Pipeline
====================================
Hardware target : 8× NVIDIA RTX A6000 (49 GB each)
Supports        : uniform, motion-based, CLIP-diversity sampling
Input           : --video_dir  OR  --video_list (combined_index.json from download_datasets.py)
Output          : frame_manifest.json  { video_id -> keyframe metadata + paths }

Why motion sampling for BLV navigation?
  Motion-high frames catch scene transitions, obstacle appearances, and activity moments —
  exactly the content a BLV user needs described. Use --method motion (default).

Usage:
  # From raw video directory
  python step1_extract_frames.py --video_dir /data/charades/videos --output_dir ./keyframes

  # From combined index (recommended after download_datasets.py)
  python step1_extract_frames.py --video_list /data/combined_index.json --output_dir ./keyframes

  # Best quality for navigation data
  python step1_extract_frames.py --video_list /data/combined_index.json \
      --method motion --n_frames 24 --output_dir ./keyframes --device cuda
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Literal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING METHODS
# ─────────────────────────────────────────────────────────────────────────────

def uniform_sample(cap: cv2.VideoCapture, n_frames: int) -> list[int]:
    """Evenly spaced frame indices across the full video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    return np.linspace(0, total - 1, min(n_frames, total), dtype=int).tolist()


def motion_based_sample(cap: cv2.VideoCapture, n_frames: int) -> list[int]:
    """
    Select frames with highest motion (frame-difference magnitude).
    Best for BLV navigation: captures obstacle appearances, scene transitions,
    and activity moments rather than static background frames.

    Strategy:
      - Sub-sample every 3rd frame for speed on long videos
      - Score by mean absolute pixel difference in grayscale
      - Always include first, last, and temporal quartile frames for coverage
      - Fill remaining slots with top-motion frames
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    if total <= n_frames:
        return list(range(total))

    # Anchor frames: first, last, and 3 temporal quartiles for scene coverage
    anchors = sorted(set([
        0,
        total // 4,
        total // 2,
        3 * total // 4,
        total - 1,
    ]))
    budget = n_frames - len(anchors)

    if budget <= 0:
        return anchors[:n_frames]

    # Score frames with a single sequential pass — no random seeks (much faster)
    # Decode every Nth frame by reading sequentially and skipping non-target frames.
    step = max(1, total // 600)  # never score more than ~600 frames for speed
    scores = []
    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                scores.append((diff, frame_idx))
            prev_gray = gray
        frame_idx += 1

    scores.sort(reverse=True)
    anchor_set = set(anchors)
    motion_frames = [idx for _, idx in scores if idx not in anchor_set][:budget]

    result = sorted(set(anchors + motion_frames))
    return result[:n_frames]


def clip_diversity_sample(cap: cv2.VideoCapture, n_frames: int, device: str = "cuda") -> list[int]:
    """
    Greedy max-diversity frame selection using CLIP embeddings.
    Selects frames that are visually maximally different from each other —
    useful for videos with many static shots.

    Candidate pool: 5× n_frames uniform candidates → greedy selection.
    Falls back to motion sampling if CLIP is unavailable.
    """
    if not CLIP_AVAILABLE:
        print("  [WARN] CLIP not available, falling back to motion sampling.")
        return motion_based_sample(cap, n_frames)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= n_frames:
        return list(range(total))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    candidates = uniform_sample(cap, n_frames * 5)

    frames_rgb = []
    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    inputs = processor(images=pil_frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings.cpu().numpy()

    # Greedy farthest-point sampling
    selected = [0]
    for _ in range(n_frames - 1):
        remaining = [i for i in range(len(candidates)) if i not in selected]
        if not remaining:
            break
        # Pick frame that maximizes minimum distance to already-selected set
        best, best_score = remaining[0], -1.0
        for r in remaining:
            min_dist = min(1.0 - float(np.dot(embeddings[r], embeddings[s])) for s in selected)
            if min_dist > best_score:
                best_score = min_dist
                best = r
        selected.append(best)

    return sorted(candidates[i] for i in selected)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE VIDEO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    video_id: str = None,
    n_frames: int = 24,
    method: Literal["uniform", "motion", "clip"] = "motion",
    device: str = "cuda",
    jpeg_quality: int = 92,
) -> dict:
    """
    Extract keyframes from a video and save as JPEG.

    Args:
        video_id: Explicit ID override; defaults to video stem.
        jpeg_quality: 92 is a good balance — higher than the previous 90,
                      preserving small obstacle details in navigation data.
    """
    video_path = Path(video_path)
    video_id = video_id or video_path.stem
    out_dir = Path(output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "error": f"Cannot open: {video_path}", "frames": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Select indices
    if method == "motion":
        indices = motion_based_sample(cap, n_frames)
    elif method == "clip":
        indices = clip_diversity_sample(cap, n_frames, device=device)
    else:
        indices = uniform_sample(cap, n_frames)

    # Save frames
    saved_paths = []
    for rank, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = out_dir / f"frame_{rank:03d}_idx{idx:06d}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved_paths.append(str(fname))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "width": width,
        "height": height,
        "method": method,
        "n_keyframes": len(saved_paths),
        "frames": saved_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _process_one(args_tuple):
    video_path, video_id, output_dir, n_frames, method, device, jpeg_quality = args_tuple
    try:
        return extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            video_id=video_id,
            n_frames=n_frames,
            method=method,
            device=device,
            jpeg_quality=jpeg_quality,
        )
    except Exception as e:
        return {"video_id": video_id or Path(video_path).stem, "error": str(e), "frames": []}


def process_dataset(
    output_dir: str,
    n_frames: int = 24,
    method: str = "motion",
    device: str = "cuda",
    manifest_path: str = "frame_manifest.json",
    resume: bool = True,
    num_workers: int = 4,
    jpeg_quality: int = 92,
    # Input sources (provide one)
    video_dir: str = None,
    video_list: str = None,          # combined_index.json from download_datasets.py
    extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
):
    """
    Process a full dataset of videos.

    Supports:
      - video_dir: scan directory recursively for video files
      - video_list: combined_index.json produced by download_datasets.py
                    (gives explicit video_id + path + dataset source metadata)

    Uses ThreadPoolExecutor for parallel I/O. n=4 workers is safe on NVMe;
    increase to 8 on fast storage.
    """
    # Build job list
    jobs = []  # (video_path, video_id)

    if video_list:
        print(f"Loading video list from {video_list}")
        with open(video_list) as f:
            index = json.load(f)
        if isinstance(index, dict):
            entries = list(index.values())
        else:
            entries = index
        for entry in entries:
            vp = entry.get("video_path", "")
            vid = entry.get("video_id", Path(vp).stem)
            if vp and Path(vp).exists():
                jobs.append((vp, vid))
            elif vp:
                print(f"  [WARN] Missing: {vp}")
    elif video_dir:
        video_dir = Path(video_dir)
        for p in video_dir.rglob("*"):
            if p.suffix.lower() in extensions:
                jobs.append((str(p), p.stem))
    else:
        raise ValueError("Provide --video_dir or --video_list")

    print(f"Found {len(jobs)} videos")

    # Resume
    existing = {}
    if resume and Path(manifest_path).exists():
        with open(manifest_path) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already done")

    jobs = [(vp, vid) for vp, vid in jobs if vid not in existing]
    print(f"Processing {len(jobs)} remaining videos with method={method}, workers={num_workers}")

    manifest = dict(existing)
    failed = []

    task_args = [
        (vp, vid, output_dir, n_frames, method, device, jpeg_quality)
        for vp, vid in jobs
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_one, a): a[1] for a in task_args}
        with tqdm(total=len(futures), desc="Extracting frames") as pbar:
            for future in as_completed(futures):
                result = future.result()
                vid = result["video_id"]
                if "error" in result:
                    print(f"  [ERROR] {vid}: {result['error']}")
                    failed.append(vid)
                else:
                    manifest[vid] = result

                pbar.update(1)

                # Checkpoint every 500 videos
                if len(manifest) % 500 == 0:
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

    # Final save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_frames = sum(v.get("n_keyframes", 0) for v in manifest.values())
    print(f"\n  Extracted : {len(manifest):,} videos, {total_frames:,} total frames")
    print(f"  Failed    : {len(failed)}")
    print(f"  Manifest  : {manifest_path}")

    if failed:
        failed_path = manifest_path.replace(".json", "_failed.json")
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"  Failures  : {failed_path}")

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Extract keyframes for BLV navigation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input (one of these)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video_dir", help="Directory to scan for videos")
    grp.add_argument(
        "--video_list",
        help="combined_index.json from download_datasets.py (recommended)",
    )

    parser.add_argument("--output_dir", default="keyframes", help="Where to save JPEG keyframes")
    parser.add_argument(
        "--n_frames", type=int, default=24,
        help="Keyframes per video. 24 = good coverage for ~30s Charades clips",
    )
    parser.add_argument(
        "--method", choices=["uniform", "motion", "clip"], default="motion",
        help="motion=best for BLV navigation, clip=max diversity, uniform=fastest",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cuda (recommended on A6000) or cpu",
    )
    parser.add_argument("--manifest", default="frame_manifest.json")
    parser.add_argument("--no_resume", action="store_true", help="Restart from scratch")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel extraction workers (4 safe on NVMe, up to 12 on fast SSD)",
    )
    parser.add_argument("--jpeg_quality", type=int, default=92)
    args = parser.parse_args()

    process_dataset(
        video_dir=args.video_dir,
        video_list=args.video_list,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        method=args.method,
        device=args.device,
        manifest_path=args.manifest,
        resume=not args.no_resume,
        num_workers=args.workers,
        jpeg_quality=args.jpeg_quality,
    )
