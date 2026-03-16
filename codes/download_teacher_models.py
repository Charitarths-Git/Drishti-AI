"""
download_teacher_models.py
==========================
Downloads 4 teacher VLM models for evaluation on video benchmarks.

Models:
  1. InternVL3-78B   — OpenGVLab/InternVL3-78B
  2. Tarsier2-7B     — omni-research/Tarsier2-7b
  3. Qwen2.5-VL-72B  — Qwen/Qwen2.5-VL-72B-Instruct
  4. Molmo-7B        — allenai/Molmo-7B-D-0924

Hardware: 8× NVIDIA RTX A6000 (49 GB each) | 20 TB storage
Storage:  ~180 GB total for all 4 models

Usage:
    # Download all models
    python download_teacher_models.py --cache_dir /data/models

    # Download specific model only
    python download_teacher_models.py --model internvl3 --cache_dir /data/models

    # Check which models are already cached
    python download_teacher_models.py --check_only

    # List model info without downloading
    python download_teacher_models.py --info_only
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

TEACHER_MODELS = {
    "internvl3": {
        "name": "InternVL3-78B",
        "hf_id": "OpenGVLab/InternVL3-78B",
        "size_params": "78B",
        "disk_gb": 156,
        "min_vram_gb": 160,    # needs multi-GPU or quantization
        "role": "Strong perception — best on PerceptionTest",
        "lmms_model_type": "internvl2",
        "notes": "Requires 4× A6000 with tensor parallel, or AWQ quantization",
    },
    "tarsier2": {
        "name": "Tarsier2-Recap-7B",
        "hf_id": "omni-research/Tarsier2-Recap-7b",
        "size_params": "7B (~8B)",
        "disk_gb": 16,
        "min_vram_gb": 16,
        "role": "Temporal reasoning — video description specialist",
        "lmms_model_type": "qwen2_5_vl",
        "notes": "Built on Qwen2-VL-7B. Gated — requires HF login. Fits 1× A6000.",
    },
    "qwen72b": {
        "name": "Qwen2.5-VL-72B",
        "hf_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "size_params": "72B",
        "disk_gb": 144,
        "min_vram_gb": 148,
        "role": "General reasoning + QA — strong on Video-MME",
        "lmms_model_type": "qwen2_5_vl",
        "notes": "Requires 4× A6000 with tensor parallel, or AWQ quantization",
    },
    "qwen7b": {
        "name": "Qwen2.5-VL-7B",
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_params": "7B",
        "disk_gb": 16,
        "min_vram_gb": 16,
        "role": "Mid-size efficient model — scaling baseline",
        "lmms_model_type": "qwen2_5_vl",
        "notes": "Already on server. Perfect scaling comparison vs 72B.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def check_disk_space(path: str, needed_gb: float) -> bool:
    """Check if enough disk space is available."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        return free_gb >= needed_gb
    except Exception:
        return True  # can't check, assume OK


def check_model_cached(hf_id: str, cache_dir: str = None) -> dict:
    """Check if a model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir, model_info

        hf_cache = cache_dir or os.environ.get("HF_HOME") or str(
            Path.home() / ".cache" / "huggingface"
        )
        info = scan_cache_dir(hf_cache)
        for repo in info.repos:
            if repo.repo_id == hf_id:
                size_gb = repo.size_on_disk / 1e9
                return {
                    "cached": True,
                    "path": str(repo.repo_path),
                    "size_gb": round(size_gb, 1),
                }
    except Exception:
        pass

    return {"cached": False, "path": None, "size_gb": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_model(
    key: str,
    model_info: dict,
    cache_dir: str = None,
    force: bool = False,
) -> dict:
    """
    Download a single teacher model using huggingface_hub.snapshot_download.
    Resumable and atomic.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface-hub")
        return {"success": False, "error": "huggingface_hub not installed"}

    hf_id = model_info["hf_id"]
    name = model_info["name"]

    print(f"\n{'─'*60}")
    print(f"  Downloading: {name}")
    print(f"  HF ID      : {hf_id}")
    print(f"  Size        : ~{model_info['disk_gb']} GB")
    print(f"  Role        : {model_info['role']}")
    print(f"{'─'*60}")

    # Check if already cached
    if not force:
        status = check_model_cached(hf_id, cache_dir)
        if status["cached"]:
            print(f"  Already cached at: {status['path']}")
            print(f"  Size on disk: {status['size_gb']:.1f} GB")
            print(f"  Use --force to re-download")
            return {"success": True, "path": status["path"], "cached": True}

    # Check disk space
    check_path = cache_dir or str(Path.home())
    if not check_disk_space(check_path, model_info["disk_gb"]):
        print(f"  [ERROR] Not enough disk space. Need ~{model_info['disk_gb']} GB")
        return {"success": False, "error": "insufficient_disk"}

    # Download
    print(f"  Starting download (this may take a while)...")
    print(f"  Safe to interrupt — download is resumable.")

    kwargs = {
        "repo_id": hf_id,
        "repo_type": "model",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    start = time.time()
    try:
        local_path = snapshot_download(**kwargs)
        elapsed = time.time() - start
        size_gb = sum(
            f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file()
        ) / 1e9

        print(f"  Downloaded to : {local_path}")
        print(f"  Size          : {size_gb:.1f} GB")
        print(f"  Time          : {elapsed/60:.1f} min")

        return {"success": True, "path": local_path, "cached": False}

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Alternative: huggingface-cli download {hf_id}")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_model(key: str, model_info: dict, model_path: str) -> bool:
    """Quick verification that the model files exist and look correct."""
    path = Path(model_path)
    if not path.exists():
        print(f"  [!!] Path does not exist: {model_path}")
        return False

    # Check for config.json (all HF models have this)
    config = path / "config.json"
    if not config.exists():
        # Try looking in subdirectories (snapshot_download layout)
        configs = list(path.rglob("config.json"))
        if not configs:
            print(f"  [!!] No config.json found in {model_path}")
            return False
        config = configs[0]

    # Check for weight files
    safetensors = list(path.rglob("*.safetensors"))
    bin_files = list(path.rglob("*.bin"))
    weight_files = safetensors + bin_files

    if not weight_files:
        print(f"  [!!] No weight files found in {model_path}")
        return False

    total_size = sum(f.stat().st_size for f in weight_files) / 1e9
    print(f"  [OK] {model_info['name']}: {len(weight_files)} weight files, {total_size:.1f} GB")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Info display
# ─────────────────────────────────────────────────────────────────────────────

def print_model_info(cache_dir: str = None):
    """Print information about all teacher models."""
    print("\n" + "═" * 70)
    print("  TEACHER MODELS FOR EVALUATION")
    print("═" * 70)

    total_disk = 0
    for key, info in TEACHER_MODELS.items():
        status = check_model_cached(info["hf_id"], cache_dir)
        cached_str = "CACHED" if status["cached"] else "NOT DOWNLOADED"
        cached_sym = "OK" if status["cached"] else "  "

        print(f"\n  [{cached_sym}] {info['name']}")
        print(f"       HF ID       : {info['hf_id']}")
        print(f"       Parameters   : {info['size_params']}")
        print(f"       Disk         : ~{info['disk_gb']} GB")
        print(f"       Min VRAM     : ~{info['min_vram_gb']} GB")
        print(f"       Role         : {info['role']}")
        print(f"       lmms-eval    : --model {info['lmms_model_type']}")
        print(f"       Status       : {cached_str}")
        if status["cached"]:
            print(f"       Path         : {status['path']}")

        total_disk += info["disk_gb"]

    print(f"\n  Total disk needed : ~{total_disk} GB")

    # Check available space
    check_path = cache_dir or str(Path.home())
    try:
        free_gb = shutil.disk_usage(check_path).free / 1e9
        print(f"  Available disk    : {free_gb:.0f} GB")
        if free_gb < total_disk:
            print(f"  [WARN] May not have enough space for all models!")
    except Exception:
        pass

    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download teacher VLM models for video benchmark evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(TEACHER_MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (or 'all')",
    )
    p.add_argument(
        "--cache_dir",
        default="/usershome/cs671_user3/models",
        help="HuggingFace model cache directory",
    )
    p.add_argument(
        "--check_only",
        action="store_true",
        help="Only check which models are cached, don't download",
    )
    p.add_argument(
        "--info_only",
        action="store_true",
        help="Print model info and exit",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if model is cached",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models have valid weight files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"HF cache dir: {args.cache_dir}")

    # Info only — print model table and exit
    if args.info_only:
        print_model_info(args.cache_dir)
        return

    # Check only — show cache status
    if args.check_only:
        print_model_info(args.cache_dir)
        return

    # Select models to download
    if args.model == "all":
        to_download = list(TEACHER_MODELS.items())
    else:
        to_download = [(args.model, TEACHER_MODELS[args.model])]

    # Show what we're about to do
    print("\n" + "═" * 60)
    print("  TEACHER MODEL DOWNLOAD")
    print("═" * 60)
    total_gb = sum(info["disk_gb"] for _, info in to_download)
    print(f"  Models to download : {len(to_download)}")
    print(f"  Total disk needed  : ~{total_gb} GB")

    for key, info in to_download:
        print(f"    • {info['name']} ({info['hf_id']})")

    print("═" * 60)

    # Download each model
    results = {}
    for key, info in to_download:
        result = download_model(key, info, args.cache_dir, force=args.force)
        results[key] = result

        # Verify if requested
        if args.verify and result.get("success") and result.get("path"):
            verify_model(key, info, result["path"])

    # Summary
    print("\n" + "═" * 60)
    print("  DOWNLOAD SUMMARY")
    print("═" * 60)

    all_ok = True
    for key, result in results.items():
        info = TEACHER_MODELS[key]
        ok = result.get("success", False)
        sym = "OK" if ok else "!!"
        status = "ready" if ok else f"FAILED: {result.get('error', 'unknown')}"
        was_cached = " (was cached)" if result.get("cached") else ""
        print(f"  [{sym}] {info['name']:25s} {status}{was_cached}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All models downloaded successfully!")
        print("\n  Next: python run_teacher_eval.py --dry_run")
    else:
        print("\n  Some downloads failed. Fix the errors above and re-run.")

    print("═" * 60)

    # Save model paths for use by run_teacher_eval.py
    model_paths = {}
    for key, result in results.items():
        if result.get("success") and result.get("path"):
            model_paths[key] = {
                "hf_id": TEACHER_MODELS[key]["hf_id"],
                "local_path": result["path"],
                "lmms_model_type": TEACHER_MODELS[key]["lmms_model_type"],
            }

    paths_file = Path(__file__).parent / "teacher_model_paths.json"
    with open(paths_file, "w") as f:
        json.dump(model_paths, f, indent=2)
    print(f"\n  Model paths saved → {paths_file}")


if __name__ == "__main__":
    main()
