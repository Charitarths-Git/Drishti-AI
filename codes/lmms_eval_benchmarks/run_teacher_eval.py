"""
run_teacher_eval.py
===================
Orchestrator — runs lmms-eval on 4 teacher models across 3 video benchmarks.

Models (4):
  InternVL3-78B, Tarsier2-7B, Qwen2.5-VL-72B, Molmo-7B

Benchmarks (3):
  EgoSchema, PerceptionTest (val MC), Video-MME

Uses --limit 400 to subsample each benchmark (instead of full 5K/1.7K/900).
Total evaluations: 4 models × 3 benchmarks = 12 runs.

Hardware: 8× NVIDIA RTX A6000 (49 GB each) | CUDA 12.1+

Usage:
    # Dry run — print all commands without executing
    python run_teacher_eval.py --dry_run

    # Run all models on all benchmarks (400 samples each)
    python run_teacher_eval.py --output_dir ./teacher_eval_results

    # Run specific model(s) only
    python run_teacher_eval.py --models molmo tarsier2

    # Run specific benchmark(s) only
    python run_teacher_eval.py --tasks egoschema videomme

    # Quick smoke test (5 samples)
    python run_teacher_eval.py --models molmo --limit 5

    # Use specific GPU(s)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_teacher_eval.py --models qwen72b

    # Resume from a partially completed run
    python run_teacher_eval.py --resume
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
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "internvl3": {
        "name": "InternVL3-78B",
        "hf_id": "OpenGVLab/InternVL3-78B",
        "lmms_model": "internvl2",
        "size_b": 78,
        "model_args": {
            "pretrained": "OpenGVLab/InternVL3-78B",
            "device_map": "auto",
            "max_num": 12,                # max frames for video
            "load_in_8bit": True,          # quantize to fit on fewer GPUs
        },
        "batch_size": 1,
        "notes": "78B model — requires multi-GPU or 8-bit quant",
    },
    "tarsier2": {
        "name": "Tarsier2-Recap-7B",
        "hf_id": "omni-research/Tarsier2-Recap-7b",
        "lmms_model": "qwen2_5_vl",
        "size_b": 7,
        "model_args": {
            "pretrained": "omni-research/Tarsier2-Recap-7b",
            "device_map": "auto",
            "max_pixels": "1003520",
            "min_pixels": "200704",
        },
        "batch_size": 1,
        "notes": "7B — built on Qwen2-VL. Gated (requires HF login). Fits 1× A6000.",
    },
    "qwen72b": {
        "name": "Qwen2.5-VL-72B",
        "hf_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "lmms_model": "qwen2_5_vl",
        "size_b": 72,
        "model_args": {
            "pretrained": "Qwen/Qwen2.5-VL-72B-Instruct",
            "device_map": "auto",
            "max_pixels": "1003520",      # 1280*28*28 optimal for Qwen-VL
            "min_pixels": "200704",        # 256*28*28
            "load_in_8bit": True,          # quantize to fit on fewer GPUs
        },
        "batch_size": 1,
        "notes": "72B model — requires multi-GPU or 8-bit quant",
    },
    "qwen7b": {
        "name": "Qwen2.5-VL-7B",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "lmms_model": "qwen2_5_vl",
        "size_b": 7,
        "model_args": {
            "pretrained": "Qwen/Qwen2.5-VL-7B-Instruct",
            "device_map": "auto",
            "max_pixels": "1003520",
            "min_pixels": "200704",
        },
        "batch_size": 1,
        "notes": "7B model. Provides perfect 7B vs 72B scaling comparison with Qwen2.5-VL-72B.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark registry
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "egoschema": {
        "name": "EgoSchema",
        "lmms_task": "egoschema",
        "full_size": 5000,
        "description": "Long-form egocentric video reasoning (3-min clips, MCQ)",
        "metrics": ["accuracy"],
    },
    "perceptiontest": {
        "name": "PerceptionTest",
        "lmms_task": "perceptiontest_val_mc",
        "full_size": 1700,
        "description": "Multimodal perception & reasoning (video MCQ)",
        "metrics": ["accuracy"],
    },
    "videomme": {
        "name": "Video-MME",
        "lmms_task": "videomme",
        "full_size": 900,
        "description": "Comprehensive video MLLM evaluation (6 domains, 30 subfields)",
        "metrics": ["accuracy"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Build lmms-eval command
# ─────────────────────────────────────────────────────────────────────────────

def build_model_args_string(model_config: dict) -> str:
    """Convert model_args dict to comma-separated key=value string for lmms-eval."""
    parts = []
    for k, v in model_config["model_args"].items():
        if isinstance(v, bool):
            parts.append(f"{k}={str(v)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ",".join(parts)


def build_eval_command(
    model_key: str,
    task_key: str,
    limit: int,
    output_dir: str,
    log_samples: bool = True,
) -> list:
    """Build the lmms-eval CLI command for one model × one benchmark."""
    model_cfg = MODELS[model_key]
    bench_cfg = BENCHMARKS[task_key]

    # Output path
    run_name = f"{model_key}_{task_key}"
    output_path = str(Path(output_dir) / run_name)

    cmd = [
        sys.executable, "-m", "lmms_eval",
        "--model", model_cfg["lmms_model"],
        "--model_args", build_model_args_string(model_cfg),
        "--tasks", bench_cfg["lmms_task"],
        "--batch_size", str(model_cfg["batch_size"]),
        "--limit", str(limit),
        "--output_path", output_path,
    ]

    if log_samples:
        cmd.append("--log_samples")

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluations
# ─────────────────────────────────────────────────────────────────────────────

def run_single_eval(
    model_key: str,
    task_key: str,
    limit: int,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """Run a single evaluation (1 model × 1 benchmark)."""
    model_cfg = MODELS[model_key]
    bench_cfg = BENCHMARKS[task_key]
    run_name = f"{model_key}_{task_key}"

    cmd = build_eval_command(model_key, task_key, limit, output_dir)

    print(f"\n{'─'*70}")
    print(f"  ▶  {model_cfg['name']}  ×  {bench_cfg['name']}")
    print(f"     Samples: {limit} / {bench_cfg['full_size']}")
    print(f"     Command:")
    print(f"     {' '.join(cmd)}")
    print(f"{'─'*70}")
    sys.stdout.flush()

    if dry_run:
        print(f"  [DRY RUN] Skipping execution")
        return {"status": "dry_run", "model": model_key, "task": task_key}

    # Check if results already exist (for resume)
    result_dir = Path(output_dir) / run_name
    if result_dir.exists():
        result_files = list(result_dir.rglob("results*.json"))
        if result_files:
            print(f"  [SKIP] Results already exist at {result_dir}")
            print(f"  Use --force to re-run")
            try:
                with open(result_files[0]) as f:
                    existing = json.load(f)
                return {
                    "status": "cached",
                    "model": model_key,
                    "task": task_key,
                    "results_path": str(result_files[0]),
                    "results": existing,
                }
            except Exception:
                pass

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=None,     # inherit stdout
            stderr=None,     # inherit stderr
            timeout=7200,    # 2 hour timeout per eval
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"  [OK] {run_name} done in {elapsed/60:.1f} min")

            # Try to parse results
            result_files = list(Path(output_dir, run_name).rglob("results*.json"))
            parsed = {}
            if result_files:
                try:
                    with open(result_files[0]) as f:
                        parsed = json.load(f)
                except Exception:
                    pass

            return {
                "status": "success",
                "model": model_key,
                "task": task_key,
                "elapsed_sec": round(elapsed, 1),
                "results_path": str(result_files[0]) if result_files else None,
                "results": parsed,
            }
        else:
            print(f"  [FAILED] {run_name} — exit code {result.returncode}")
            return {
                "status": "failed",
                "model": model_key,
                "task": task_key,
                "exit_code": result.returncode,
                "elapsed_sec": round(elapsed, 1),
            }

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {run_name} — exceeded 2 hour limit")
        return {
            "status": "timeout",
            "model": model_key,
            "task": task_key,
        }
    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        return {
            "status": "error",
            "model": model_key,
            "task": task_key,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_all_evaluations(
    models: list,
    tasks: list,
    limit: int,
    output_dir: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Run all model × benchmark evaluations."""
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_runs = len(models) * len(tasks)

    print("\n" + "═" * 70)
    print("  TEACHER MODEL EVALUATION — lmms-eval")
    print("═" * 70)
    print(f"  Models     : {', '.join(MODELS[m]['name'] for m in models)}")
    print(f"  Benchmarks : {', '.join(BENCHMARKS[t]['name'] for t in tasks)}")
    print(f"  Limit      : {limit} samples per benchmark")
    print(f"  Total runs : {total_runs}")
    print(f"  Output     : {output_dir}/")
    print(f"  Mode       : {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 70)

    # Estimate time: ~5 min per eval for 7B models, ~20 min for 72B+
    est_minutes = sum(
        5 if MODELS[m]["size_b"] < 20 else 20
        for m in models
        for _ in tasks
    )
    print(f"\n  Estimated time: ~{est_minutes} minutes ({est_minutes/60:.1f} hours)")

    all_results = {}
    completed = 0
    failed = 0

    # Order: run smaller models first (faster, test pipeline)
    ordered_models = sorted(models, key=lambda m: MODELS[m]["size_b"])

    for model_key in ordered_models:
        for task_key in tasks:
            completed += 1
            print(f"\n  [{completed}/{total_runs}] ", end="")

            result = run_single_eval(
                model_key=model_key,
                task_key=task_key,
                limit=limit,
                output_dir=output_dir,
                dry_run=dry_run,
            )

            run_name = f"{model_key}_{task_key}"
            all_results[run_name] = result

            if result["status"] in ("failed", "error", "timeout"):
                failed += 1

    # Save combined results
    results_file = Path(output_dir) / "teacher_eval_all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    _print_summary(all_results, models, tasks, output_dir)

    return all_results


def _print_summary(all_results: dict, models: list, tasks: list, output_dir: str):
    """Print a summary table of all evaluation results."""
    print("\n" + "═" * 70)
    print("  EVALUATION SUMMARY")
    print("═" * 70)

    # Header
    task_names = [BENCHMARKS[t]["name"] for t in tasks]
    header = f"  {'Model':<25s}" + "".join(f"  {t:<18s}" for t in task_names) + "  Status"
    print(header)
    print("  " + "─" * 66)

    for model_key in models:
        model_name = MODELS[model_key]["name"]
        row = f"  {model_name:<25s}"

        for task_key in tasks:
            run_name = f"{model_key}_{task_key}"
            result = all_results.get(run_name, {})
            status = result.get("status", "?")

            if status in ("success", "cached"):
                # Try to extract the accuracy score
                results_data = result.get("results", {})
                score = _extract_score(results_data, task_key)
                if score is not None:
                    row += f"  {score:>6.1f}%{'':>11s}"
                else:
                    row += f"  {'done':>18s}"
            elif status == "dry_run":
                row += f"  {'(dry run)':>18s}"
            else:
                row += f"  {'FAILED':>18s}"

        # Overall status
        statuses = [all_results.get(f"{model_key}_{t}", {}).get("status", "?") for t in tasks]
        if all(s in ("success", "cached") for s in statuses):
            row += "  ✓"
        elif all(s == "dry_run" for s in statuses):
            row += "  (dry)"
        else:
            row += "  ✗"

        print(row)

    print("═" * 70)

    # Timing
    total_time = sum(
        r.get("elapsed_sec", 0)
        for r in all_results.values()
        if r.get("status") in ("success",)
    )
    if total_time > 0:
        print(f"\n  Total evaluation time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")

    print(f"  Results saved → {Path(output_dir) / 'teacher_eval_all_results.json'}")
    print(f"\n  Next: python analyze_teacher_results.py --results_dir {output_dir}")


def _extract_score(results_data: dict, task_key: str) -> float:
    """Try to extract accuracy score from lmms-eval results JSON."""
    if not results_data:
        return None

    # lmms-eval results structure: {"results": {"task_name": {"metric_name": value}}}
    results = results_data.get("results", results_data)

    bench = BENCHMARKS.get(task_key, {})
    task_name = bench.get("lmms_task", task_key)

    # Try direct lookup
    for key in [task_name, task_key]:
        if key in results:
            task_results = results[key]
            for metric in ["accuracy", "acc", "exact_match", "score"]:
                if metric in task_results:
                    val = task_results[metric]
                    # lmms-eval often returns 0-1, convert to percentage
                    return val * 100 if val <= 1.0 else val

    # Try nested search
    for key, val in results.items():
        if isinstance(val, dict):
            for metric in ["accuracy", "acc", "exact_match", "score"]:
                if metric in val:
                    score = val[metric]
                    return score * 100 if score <= 1.0 else score

    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run lmms-eval on teacher models for video benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Models to evaluate",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=list(BENCHMARKS.keys()),
        default=list(BENCHMARKS.keys()),
        help="Benchmarks to run",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Number of samples per benchmark (400 from ~12K total)",
    )
    p.add_argument(
        "--output_dir",
        default="./teacher_eval_results",
        help="Directory to save lmms-eval outputs",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results exist",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip evaluations that already have results",
    )

    return p.parse_args()


def main():
    args = parse_args()

    results = run_all_evaluations(
        models=args.models,
        tasks=args.tasks,
        limit=args.limit,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
