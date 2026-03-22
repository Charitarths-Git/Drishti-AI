"""
BLV Navigation Dataset Pipeline — Full Orchestrator
=====================================================
Hardware target : 8× NVIDIA RTX A6000 (49 GB each) | 20 TB storage
Datasets        : AVCaps (2,061 videos) + Charades original (9,848 videos) = ~11,909 videos
Target output   : ~240K samples at 20 pairs/video, ~480K at 40 pairs/video

Default stack (fully local, zero API cost):
  Step 1: Motion-based frame extraction (24 frames/video)
  Step 2: Qwen2.5-VL-7B-Instruct — scene summaries, distributed 8× A6000
  Step 3: Qwen2.5-7B-Instruct    — BLV QA generation, distributed 8× A6000
  Step 4: SmolVLA + JSONL + HuggingFace output formats

Usage:
  # Full pipeline, fully local (recommended)
  python run_pipeline.py \\
      --video_list /data/combined_index.json \\
      --output_root ./pipeline_output \\
      --n_pairs 20 --distributed

  # Estimate cost/time without running
  python run_pipeline.py \\
      --video_list /data/combined_index.json \\
      --output_root ./pipeline_output --estimate_only

  # Skip already-done steps (e.g., resume after Step 2)
  python run_pipeline.py \\
      --video_list /data/combined_index.json \\
      --output_root ./pipeline_output \\
      --skip_steps 1 2

  # Override: use GPT-4o for scene summaries, keep local for query gen
  python run_pipeline.py \\
      --video_dir /data/videos \\
      --output_root ./pipeline_output \\
      --vlm_backend gpt4o --api_key sk-...
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

# Measured on 8× A6000 with Qwen2.5-VL-7B + flash-attn + bfloat16
STEP2_SEC_PER_VIDEO_PER_GPU = 4.5   # scene summary inference + decode, 12 frames
STEP3_SEC_PER_VIDEO_PER_GPU = 8.0   # 3 prompt calls (group A/B/C) + quality filter

# GPT-4o pricing (2025)
GPT4O_IN_PER_1K  = 0.0025
GPT4O_OUT_PER_1K = 0.010
GPT4O_MINI_IN    = 0.00015
GPT4O_MINI_OUT   = 0.00060
IMG_TOKENS_LOW   = 85    # "low" detail mode per image
SCENE_OUT_TOKENS = 600
QUERY_IN_TOKENS  = 800
QUERY_OUT_TOKENS = 2400


def estimate_pipeline(
    n_videos: int,
    n_frames_vlm: int = 12,
    n_pairs: int = 20,
    n_gpus: int = 8,
    vlm_backend: str = "local",
    query_backend: str = "local",
    openai_vlm_model: str = "gpt-4o",
    openai_query_model: str = "gpt-4o-mini",
) -> dict:
    # Step 2 cost
    if vlm_backend == "gpt4o":
        s2_in = n_videos * (n_frames_vlm * IMG_TOKENS_LOW + 350)
        s2_out = n_videos * SCENE_OUT_TOKENS
        s2_cost = s2_in / 1000 * GPT4O_IN_PER_1K + s2_out / 1000 * GPT4O_OUT_PER_1K
    else:
        s2_cost = 0.0

    # Step 3 cost (3 calls per video for group A/B/C + quality check)
    if query_backend == "openai":
        model_in = GPT4O_IN_PER_1K if "mini" not in openai_query_model else GPT4O_MINI_IN
        model_out = GPT4O_OUT_PER_1K if "mini" not in openai_query_model else GPT4O_MINI_OUT
        s3_in = n_videos * QUERY_IN_TOKENS * 4   # 4 calls per video
        s3_out = n_videos * QUERY_OUT_TOKENS * 4
        s3_cost = s3_in / 1000 * model_in + s3_out / 1000 * model_out
    else:
        s3_cost = 0.0

    # Time estimates
    s2_hours = (n_videos / n_gpus * STEP2_SEC_PER_VIDEO_PER_GPU) / 3600
    s3_hours = (n_videos / n_gpus * STEP3_SEC_PER_VIDEO_PER_GPU) / 3600

    return {
        "n_videos": n_videos,
        "n_pairs": n_pairs,
        "total_samples_estimated": n_videos * n_pairs,
        "n_gpus": n_gpus,
        "step2_cost_usd": round(s2_cost, 2),
        "step3_cost_usd": round(s3_cost, 2),
        "total_cost_usd": round(s2_cost + s3_cost, 2),
        "step2_hours_estimated": round(s2_hours, 1),
        "step3_hours_estimated": round(s3_hours, 1),
        "total_hours_estimated": round(s2_hours + s3_hours, 1),
    }


def print_estimate(e: dict):
    print("\n" + "═" * 60)
    print("  PIPELINE ESTIMATE")
    print("═" * 60)
    print(f"  Videos           : {e['n_videos']:>8,}")
    print(f"  Pairs/video      : {e['n_pairs']:>8,}")
    print(f"  Estimated samples: {e['total_samples_estimated']:>8,}")
    print(f"  GPUs             : {e['n_gpus']:>8}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Step 2 (VLM)     : ${e['step2_cost_usd']:>7.2f}   ~{e['step2_hours_estimated']:.1f}h")
    print(f"  Step 3 (query)   : ${e['step3_cost_usd']:>7.2f}   ~{e['step3_hours_estimated']:.1f}h")
    print(f"  TOTAL COST       : ${e['total_cost_usd']:>7.2f}   ~{e['total_hours_estimated']:.1f}h wall-clock")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_step(description: str, script: str, step_args: list[str], dry_run: bool = False) -> bool:
    cmd = [sys.executable, script] + step_args
    print(f"\n{'─'*60}")
    print(f"  ▶  {description}")
    print(f"     {' '.join(cmd)}")
    print(f"{'─'*60}")
    sys.stdout.flush()

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    start = time.time()
    # Run with stdout/stderr inherited so errors print in real time
    result = subprocess.run(cmd, check=False, stdout=None, stderr=None)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  [OK] Done in {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    else:
        print(f"\n{'!'*60}")
        print(f"  [FAILED] {description}")
        print(f"  Exit code : {result.returncode}")
        print(f"  Command   : {' '.join(cmd)}")
        print(f"  Elapsed   : {elapsed:.0f}s")
        print(f"{'!'*60}")
        print("  Check the error output above for the root cause.")
        sys.stdout.flush()

    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    output_root: str,
    # Input
    video_dir: str = None,
    video_list: str = None,
    combined_index: str = None,
    # Step 1
    n_frames: int = 24,
    frame_method: str = "motion",
    frame_workers: int = 4,
    # Step 2
    vlm_backend: str = "local",       # "local" or "gpt4o"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_frames_vlm: int = 12,
    # Step 3
    query_backend: str = "local",     # "local" or "openai"
    query_model: str = "Qwen/Qwen2.5-7B-Instruct",
    openai_query_model: str = "gpt-4o-mini",
    n_pairs: int = 20,
    quality_filter: bool = True,
    # GPU
    distributed: bool = True,
    n_gpus: int = 8,
    # Auth
    api_key: str = None,
    # Output
    output_formats: list = None,
    min_quality: int = 2,
    # Control
    dry_run: bool = False,
    skip_steps: list = None,
    estimate_only: bool = False,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    skip_steps = [str(s) for s in (skip_steps or [])]
    output_formats = output_formats or ["jsonl", "smolvla", "hf"]

    pipeline_dir = Path(__file__).parent

    # Paths
    manifest    = str(output_root / "frame_manifest.json")
    summaries   = str(output_root / "scene_summaries.json")
    queries_out = str(output_root / "blv_queries.json")
    training    = str(output_root / "training_data")

    # Determine video input arg for Step 1
    if video_list:
        step1_input_args = ["--video_list", video_list]
        # Count for estimation
        with open(video_list) as f:
            index = json.load(f)
        n_videos = len(index)
    elif video_dir:
        step1_input_args = ["--video_dir", video_dir]
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        n_videos = sum(1 for p in Path(video_dir).rglob("*") if p.suffix.lower() in exts)
    else:
        raise ValueError("Provide --video_list or --video_dir")

    # Infer combined_index if not specified
    if not combined_index:
        if video_list:
            candidate = str(Path(video_list).parent / "combined_index.json")
            if Path(candidate).exists():
                combined_index = candidate

    # Estimate
    estimate = estimate_pipeline(
        n_videos=n_videos,
        n_frames_vlm=max_frames_vlm,
        n_pairs=n_pairs,
        n_gpus=n_gpus if distributed else 1,
        vlm_backend="gpt4o" if vlm_backend == "gpt4o" else "local",
        query_backend="openai" if query_backend == "openai" else "local",
        openai_query_model=openai_query_model,
    )
    print_estimate(estimate)

    if estimate_only:
        return estimate

    start_time = time.time()
    print(f"Pipeline start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output root   : {output_root}/")

    # ── STEP 1: Frame Extraction ────────────────────────────────────────────
    if "1" not in skip_steps:
        ok = run_step(
            description="Step 1: Extract keyframes (motion-based)",
            script=str(pipeline_dir / "step1_extract_frames.py"),
            step_args=step1_input_args + [
                "--output_dir", str(output_root / "keyframes"),
                "--n_frames",   str(n_frames),
                "--method",     frame_method,
                "--manifest",   manifest,
                "--device",     "cuda",
                "--workers",    str(frame_workers),
            ],
            dry_run=dry_run,
        )
        if not ok and not dry_run:
            print("Pipeline aborted at Step 1.")
            return None

    # ── STEP 2: Scene Summaries ─────────────────────────────────────────────
    if "2" not in skip_steps:
        s2_args = [
            "--manifest",   manifest,
            "--output",     summaries,
            "--max_frames", str(max_frames_vlm),
        ]

        if vlm_backend == "local":
            s2_args += ["--model", vlm_model]
            if distributed:
                s2_args += ["--distributed", "--n_gpus", str(n_gpus)]
        else:
            # GPT-4o fallback
            s2_args += ["--backend", "gpt4o"]
            if api_key:
                s2_args += ["--api_key", api_key]

        if combined_index:
            s2_args += ["--combined_index", combined_index]

        ok = run_step(
            description=f"Step 2: Scene summaries ({vlm_model if vlm_backend == 'local' else 'GPT-4o'})",
            script=str(pipeline_dir / "step2_scene_summary.py"),
            step_args=s2_args,
            dry_run=dry_run,
        )
        if not ok and not dry_run:
            print("\nPipeline aborted at Step 2. Fix the error above and re-run with --skip_steps 1")
            sys.exit(1)

    # ── STEP 3: BLV Query Generation ────────────────────────────────────────
    if "3" not in skip_steps:
        s3_args = [
            "--summaries", summaries,
            "--output",    queries_out,
            "--n_pairs",   str(n_pairs),
            "--backend",   query_backend,
        ]

        if query_backend == "local":
            s3_args += ["--local_model", query_model]
            if distributed:
                s3_args += ["--distributed", "--n_gpus", str(n_gpus)]
        else:
            s3_args += ["--openai_model", openai_query_model]
            if api_key:
                s3_args += ["--api_key", api_key]

        if not quality_filter:
            s3_args += ["--no_quality_filter"]

        if combined_index:
            s3_args += ["--combined_index", combined_index]

        ok = run_step(
            description=f"Step 3: BLV query generation ({query_model if query_backend == 'local' else openai_query_model})",
            script=str(pipeline_dir / "step3_generate_queries.py"),
            step_args=s3_args,
            dry_run=dry_run,
        )
        if not ok and not dry_run:
            print("\nPipeline aborted at Step 3. Fix the error above and re-run with --skip_steps 1 2")
            sys.exit(1)

    # ── STEP 4: Format Conversion ────────────────────────────────────────────
    if "4" not in skip_steps:
        s4_args = [
            "--queries",     queries_out,
            "--manifest",    manifest,
            "--summaries",   summaries,
            "--output_dir",  training,
            "--formats",     *output_formats,
            "--max_frames",  "8",
            "--min_quality", str(min_quality),
        ]
        run_step(
            description="Step 4: Convert to SmolVLA training format",
            script=str(pipeline_dir / "step4_convert_format.py"),
            step_args=s4_args,
            dry_run=dry_run,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  Total wall-clock time : {elapsed/3600:.2f}h")
    print(f"  Output root           : {output_root}/")
    print(f"  Frame manifest        : {manifest}")
    print(f"  Scene summaries       : {summaries}")
    print(f"  BLV queries           : {queries_out}")
    print(f"  Training data         : {training}/")
    print(f"  Estimated samples     : ~{n_videos * n_pairs:,}")
    print(f"  Finished at           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}\n")

    return estimate


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLV Navigation Dataset Pipeline — AVCaps + Charades → SmolVLA training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--video_list",
                     help="combined_index.json from download_datasets.py (recommended)")
    inp.add_argument("--video_dir",
                     help="Raw video directory (scanned recursively)")

    parser.add_argument("--output_root", default="pipeline_output")
    parser.add_argument("--combined_index", default=None,
                        help="Explicitly pass combined_index.json (auto-detected if --video_list used)")

    # Step 1
    g1 = parser.add_argument_group("Step 1: Frame extraction")
    g1.add_argument("--n_frames", type=int, default=24)
    g1.add_argument("--frame_method", choices=["uniform", "motion", "clip"], default="motion",
                    help="motion=best for BLV navigation data")
    g1.add_argument("--frame_workers", type=int, default=4)

    # Step 2
    g2 = parser.add_argument_group("Step 2: Scene summaries (VLM teacher)")
    g2.add_argument("--vlm_backend", choices=["local", "gpt4o"], default="local",
                    help="local=Qwen2.5-VL-7B (free), gpt4o=API")
    g2.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    g2.add_argument("--max_frames_vlm", type=int, default=12,
                    help="Frames per video sent to VLM (12 optimal for 7B)")

    # Step 3
    g3 = parser.add_argument_group("Step 3: BLV query generation")
    g3.add_argument("--query_backend", choices=["local", "openai"], default="local",
                    help="local=Qwen2.5-7B (free), openai=API")
    g3.add_argument("--query_model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Local LLM for query generation")
    g3.add_argument("--openai_query_model", default="gpt-4o-mini")
    g3.add_argument("--n_pairs", type=int, default=20,
                    help="QA pairs per video. 20=240K total, 40=480K total")
    g3.add_argument("--no_quality_filter", action="store_true",
                    help="Skip LLM quality scoring (faster, ~10% lower quality)")

    # GPU
    g4 = parser.add_argument_group("GPU / distributed")
    g4.add_argument("--distributed", action="store_true",
                    help="Distribute across all --n_gpus (strongly recommended)")
    g4.add_argument("--n_gpus", type=int, default=8)

    # Auth
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")

    # Output
    parser.add_argument(
        "--formats", nargs="+",
        choices=["jsonl", "hf", "smolvla"],
        default=["jsonl", "smolvla", "hf"],
    )
    parser.add_argument("--min_quality", type=int, default=2,
                        help="Min quality score for final dataset (0=all, 2=Good+)")

    # Control
    parser.add_argument("--skip_steps", nargs="*", default=[],
                        help="Steps to skip, e.g.: --skip_steps 1 2")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print all commands without executing")
    parser.add_argument("--estimate_only", action="store_true",
                        help="Show cost/time estimate and exit")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    run_pipeline(
        output_root=args.output_root,
        video_dir=args.video_dir,
        video_list=args.video_list,
        combined_index=args.combined_index,
        n_frames=args.n_frames,
        frame_method=args.frame_method,
        frame_workers=args.frame_workers,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        max_frames_vlm=args.max_frames_vlm,
        query_backend=args.query_backend,
        query_model=args.query_model,
        openai_query_model=args.openai_query_model,
        n_pairs=args.n_pairs,
        quality_filter=not args.no_quality_filter,
        distributed=args.distributed,
        n_gpus=args.n_gpus,
        api_key=api_key,
        output_formats=args.formats,
        min_quality=args.min_quality,
        dry_run=args.dry_run,
        skip_steps=args.skip_steps,
        estimate_only=args.estimate_only,
    )
