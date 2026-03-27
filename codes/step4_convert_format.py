"""
Step 4 — Convert to SmolVLA / HuggingFace Instruction-Tuning Format
====================================================================
Input  : blv_queries.json    (from Step 3)
         frame_manifest.json (from Step 1)
         scene_summaries.json (from Step 2, optional — enriches metadata)
Output : JSONL flat file, SmolVLA sharded JSONL, or HuggingFace datasets format

SmolVLA training sample format:
{
  "id": "charades__ABC123_000042",
  "video_id": "charades__ABC123",
  "images": ["keyframes/ABC123/frame_000.jpg", ...],   # up to 8 frames
  "conversations": [
    {"from": "human", "value": "Can I walk forward safely?"},
    {"from": "gpt",   "value": "No. A chair is at 12 o'clock, 0.8 meters ahead..."}
  ],
  "category": "obstacle_detection",
  "quality_score": 3,
  "metadata": {
    "dataset": "charades",
    "scene_type": "indoor_home",
    "duration_sec": 28.4,
    "lighting": "bright_artificial",
    "floor_surface": "hardwood",
    "crowd_density": "sparse",
    "had_avcaps_audio": false
  }
}

Usage:
  python step4_convert_format.py \\
      --queries blv_queries.json \\
      --manifest frame_manifest.json \\
      --summaries scene_summaries.json \\
      --output_dir training_data \\
      --formats jsonl smolvla hf

  # With quality threshold (keep only quality_score >= 2)
  python step4_convert_format.py \\
      --queries blv_queries.json --manifest frame_manifest.json \\
      --min_quality 2 --output_dir training_data
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CORE CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def build_training_samples(
    queries_path: str,
    manifest_path: str,
    summaries_path: str = None,
    max_frames_per_sample: int = 8,
    min_quality: int = 0,
) -> list[dict]:
    """
    Merge queries + frame manifest + scene summaries into flat training samples.

    Args:
        max_frames_per_sample: SmolVLA works best with 6-8 frames. 8 = full context.
        min_quality: Filter out pairs with quality_score < this value.
                     0 = keep all (default), 2 = keep Good+Excellent only.
        summaries_path: Optional — pulls scene_type, lighting, floor_surface etc.
                        from Step 2 outputs to enrich sample metadata.
    """
    with open(queries_path) as f:
        queries = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    summaries = {}
    if summaries_path and Path(summaries_path).exists():
        with open(summaries_path) as f:
            summaries = json.load(f)
        print(f"Loaded scene summaries: {len(summaries):,} entries")

    samples = []
    skipped_quality = 0
    skipped_no_frames = 0

    for video_id, pairs in tqdm(queries.items(), desc="Building samples"):
        if not pairs:
            continue

        video_meta = manifest.get(video_id, {})
        frames = video_meta.get("frames", [])

        if not frames:
            skipped_no_frames += 1
            continue

        # Sub-sample frames evenly to max_frames_per_sample
        if len(frames) > max_frames_per_sample:
            step = len(frames) // max_frames_per_sample
            frames_for_sample = frames[::step][:max_frames_per_sample]
        else:
            frames_for_sample = frames

        # Only keep frames that actually exist on disk
        frames_for_sample = [f for f in frames_for_sample if Path(f).exists()]
        if not frames_for_sample:
            skipped_no_frames += 1
            continue

        # Pull scene metadata from summaries
        summary_entry = summaries.get(video_id, {})
        scene = summary_entry.get("scene", {})
        scene_meta = {
            "dataset": summary_entry.get("dataset", video_meta.get("dataset", "unknown")),
            "scene_type": scene.get("scene_type"),
            "lighting": scene.get("lighting"),
            "floor_surface": scene.get("floor_surface"),
            "crowd_density": scene.get("crowd_density"),
            "had_avcaps_audio": summary_entry.get("had_avcaps_audio", False),
            "duration_sec": video_meta.get("duration_sec"),
            "fps": video_meta.get("fps"),
            "extraction_method": video_meta.get("method"),
            "n_keyframes_total": video_meta.get("n_keyframes"),
        }

        for i, pair in enumerate(pairs):
            # Quality filter
            qs = pair.get("quality_score", 2)
            if qs < min_quality:
                skipped_quality += 1
                continue

            # Generate a stable unique ID
            pair_hash = abs(hash(pair.get("question", "") + video_id)) % 10_000_000
            sample_id = f"{video_id}_{pair_hash:07d}"

            sample = {
                "id": sample_id,
                "video_id": video_id,
                "images": frames_for_sample,
                "conversations": [
                    {"from": "human", "value": pair["question"]},
                    {"from": "gpt",   "value": pair["answer"]},
                ],
                "category": pair.get("category", "scene_understanding"),
                "quality_score": qs,
                "metadata": scene_meta,
            }
            samples.append(sample)

    print(f"  Built          : {len(samples):,} samples")
    print(f"  Skipped (quality filter) : {skipped_quality:,}")
    print(f"  Skipped (missing frames) : {skipped_no_frames:,}")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMAT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(samples: list[dict], output_path: str):
    """Flat JSONL — one sample per line. Most compatible format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  JSONL: {output_path}  ({len(samples):,} samples, {size_mb:.1f} MB)")


def write_smolvla_format(samples: list[dict], output_dir: str, shard_size: int = 10_000):
    """
    Write sharded JSONL for SmolVLA training scripts.
    Each file: smolvla_train_000000.jsonl  (10K samples per shard by default)

    Shard naming is zero-padded to 6 digits for easy sorting and globbing:
      ls smolvla_shards/smolvla_train_*.jsonl | wc -l
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_shards = max(1, (len(samples) + shard_size - 1) // shard_size)

    for shard_idx in range(n_shards):
        shard = samples[shard_idx * shard_size: (shard_idx + 1) * shard_size]
        shard_path = Path(output_dir) / f"smolvla_train_{shard_idx:06d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for s in shard:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_size_mb = sum(
        p.stat().st_size for p in Path(output_dir).glob("smolvla_train_*.jsonl")
    ) / 1e6
    print(f"  SmolVLA shards: {output_dir}/  ({n_shards} shards, {len(samples):,} samples, {total_size_mb:.1f} MB)")


def write_hf_dataset(
    samples: list[dict],
    output_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.01,
):
    """
    Write HuggingFace datasets format with train/validation/test splits.
    Stratified by category to ensure all categories appear in all splits.
    """
    from datasets import Dataset, DatasetDict

    # Group by category for stratified split
    by_cat = {}
    for s in samples:
        cat = s.get("category", "unknown")
        by_cat.setdefault(cat, []).append(s)

    train_list, val_list, test_list = [], [], []
    for cat, cat_samples in by_cat.items():
        random.shuffle(cat_samples)
        n = len(cat_samples)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        test_list.extend(cat_samples[:n_test])
        val_list.extend(cat_samples[n_test: n_test + n_val])
        train_list.extend(cat_samples[n_test + n_val:])

    splits = {"train": train_list, "validation": val_list, "test": test_list}

    # HuggingFace datasets needs list-type columns to be consistent
    # Flatten conversations to instruction/response for simpler loading
    def flatten(s):
        convs = s.get("conversations", [])
        return {
            **{k: v for k, v in s.items() if k != "conversations"},
            "instruction": convs[0]["value"] if len(convs) > 0 else "",
            "response":    convs[1]["value"] if len(convs) > 1 else "",
        }

    ds_dict = DatasetDict({
        k: Dataset.from_list([flatten(s) for s in v])
        for k, v in splits.items()
    })
    ds_dict.save_to_disk(output_dir)

    print(f"  HuggingFace dataset: {output_dir}/")
    for split_name, split_data in splits.items():
        print(f"    {split_name:12s}: {len(split_data):,} samples")


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(samples: list[dict]):
    if not samples:
        print("  No samples.")
        return

    print(f"\n  Dataset Statistics")
    print(f"  {'─'*50}")
    print(f"  Total samples        : {len(samples):,}")

    unique_videos = len(set(s["video_id"] for s in samples))
    print(f"  Unique videos        : {unique_videos:,}")
    print(f"  Avg pairs/video      : {len(samples) / max(1, unique_videos):.1f}")

    # Dataset source breakdown
    dataset_counts = {}
    for s in samples:
        ds = s.get("metadata", {}).get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    print(f"\n  Source dataset:")
    for ds, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        print(f"    {ds:20s} {count:7,}  ({count/len(samples)*100:.1f}%)")

    # Category breakdown
    cat_counts = {}
    for s in samples:
        cat = s.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"\n  Category breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:30s} {count:7,}  ({count/len(samples)*100:.1f}%)")

    # Quality score distribution
    qs_counts = {}
    for s in samples:
        qs = s.get("quality_score", "n/a")
        qs_counts[qs] = qs_counts.get(qs, 0) + 1
    if any(isinstance(k, int) for k in qs_counts):
        print(f"\n  Quality scores:")
        for score in sorted(k for k in qs_counts if isinstance(k, int)):
            label = {3: "Excellent", 2: "Good", 1: "Poor", 0: "Invalid"}.get(score, "?")
            print(f"    {score} ({label:9s}) : {qs_counts[score]:7,}")

    # Instruction + response length stats
    inst_lens = [len(s["conversations"][0]["value"]) for s in samples if s.get("conversations")]
    resp_lens = [len(s["conversations"][1]["value"]) for s in samples if len(s.get("conversations", [])) > 1]
    if inst_lens:
        print(f"\n  Instruction length   : min={min(inst_lens)}, avg={sum(inst_lens)//len(inst_lens)}, max={max(inst_lens)}")
    if resp_lens:
        print(f"  Response length      : min={min(resp_lens)}, avg={sum(resp_lens)//len(resp_lens)}, max={max(resp_lens)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def convert_dataset(
    queries_path: str,
    manifest_path: str,
    output_dir: str = "training_data",
    summaries_path: str = None,
    formats: list[str] = None,
    max_frames: int = 8,
    min_quality: int = 0,
    val_ratio: float = 0.05,
    test_ratio: float = 0.01,
    shuffle_seed: int = 42,
    shard_size: int = 10_000,
):
    if formats is None:
        formats = ["jsonl", "smolvla"]

    random.seed(shuffle_seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Building training samples...")
    samples = build_training_samples(
        queries_path=queries_path,
        manifest_path=manifest_path,
        summaries_path=summaries_path,
        max_frames_per_sample=max_frames,
        min_quality=min_quality,
    )

    random.shuffle(samples)
    print_dataset_stats(samples)

    print(f"\nWriting outputs to {output_dir}/")

    if "jsonl" in formats:
        write_jsonl(samples, str(Path(output_dir) / "dataset.jsonl"))

    if "smolvla" in formats:
        write_smolvla_format(
            samples,
            str(Path(output_dir) / "smolvla_shards"),
            shard_size=shard_size,
        )

    if "hf" in formats:
        write_hf_dataset(
            samples,
            str(Path(output_dir) / "hf_dataset"),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    print(f"\n  Conversion complete.")
    print(f"  Output directory: {output_dir}/")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Convert to SmolVLA training format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", required=True, help="blv_queries.json from Step 3")
    parser.add_argument("--manifest", required=True, help="frame_manifest.json from Step 1")
    parser.add_argument("--summaries", default=None,
                        help="scene_summaries.json from Step 2 (enriches metadata)")
    parser.add_argument("--output_dir", default="training_data")
    parser.add_argument(
        "--formats", nargs="+", choices=["jsonl", "hf", "smolvla"],
        default=["jsonl", "smolvla"],
    )
    parser.add_argument("--max_frames", type=int, default=8,
                        help="Max frames per training sample (6-8 recommended for SmolVLA)")
    parser.add_argument("--min_quality", type=int, default=0,
                        help="Minimum quality score to include (0=all, 2=Good+Excellent only)")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=10_000,
                        help="Samples per SmolVLA shard file")
    args = parser.parse_args()

    convert_dataset(
        queries_path=args.queries,
        manifest_path=args.manifest,
        summaries_path=args.summaries,
        output_dir=args.output_dir,
        formats=args.formats,
        max_frames=args.max_frames,
        min_quality=args.min_quality,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle_seed=args.seed,
        shard_size=args.shard_size,
    )
