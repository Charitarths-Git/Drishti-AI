"""
setup_qwen.py — Qwen2.5 Local Setup, Verification & Benchmark
==============================================================
Hardware target : 8× NVIDIA RTX A6000 (49 GB each) | CUDA 12.1+

This script:
  1. Checks system requirements (CUDA, VRAM, flash-attn, packages)
  2. Downloads Qwen2.5-VL-7B-Instruct  (vision model for Step 2)
  3. Downloads Qwen2.5-7B-Instruct     (text model for Step 3)
  4. Runs inference tests on both models
  5. Benchmarks throughput on your hardware
  6. Prints a ready-to-use run command

Usage:
  # Full setup + verification (recommended first run)
  python setup_qwen.py

  # Check environment only (no downloads)
  python setup_qwen.py --check_only

  # Download models only, skip inference test
  python setup_qwen.py --download_only

  # Benchmark throughput across all 8 GPUs
  python setup_qwen.py --benchmark --n_gpus 8

  # Use a custom cache directory (default: ~/.cache/huggingface)
  python setup_qwen.py --cache_dir /data/models
"""

import os
import sys
import json
import time
import argparse
import subprocess
import platform
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "vlm": {
        "name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "role": "Step 2 — scene summaries (vision)",
        "vram_gb": 16,
        "disk_gb": 16,
    },
    "llm": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "role": "Step 3 — BLV query generation (text)",
        "vram_gb": 15,
        "disk_gb": 15,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_system() -> dict:
    results = {}

    print("=" * 60)
    print("  SYSTEM CHECK")
    print("=" * 60)

    # Python version
    pv = sys.version_info
    ok = pv >= (3, 10)
    results["python"] = ok
    _status(f"Python {pv.major}.{pv.minor}.{pv.micro}", ok, "need >= 3.10")

    # PyTorch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        results["torch"] = True
        results["cuda"] = cuda_ok
        _status(f"PyTorch {torch.__version__}", True)
        _status(f"CUDA available", cuda_ok, "GPU inference will not work without CUDA")

        if cuda_ok:
            n_gpus = torch.cuda.device_count()
            results["n_gpus"] = n_gpus
            _status(f"GPUs detected: {n_gpus}", n_gpus > 0)

            print()
            total_vram = 0
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                total_vram += vram_gb
                compute = f"{props.major}.{props.minor}"
                ampere_ok = props.major >= 8  # Ampere = compute 8.x+
                fa2_note = "flash-attn compatible" if ampere_ok else "flash-attn not supported"
                print(f"    GPU {i}: {props.name}  {vram_gb:.0f} GB VRAM  compute {compute}  [{fa2_note}]")

            results["total_vram_gb"] = total_vram
            results["ampere"] = torch.cuda.get_device_properties(0).major >= 8
            print()
            _status(f"Total VRAM: {total_vram:.0f} GB", total_vram >= 16,
                    f"need >=16 GB for one model (you have {total_vram:.0f} GB)")
    except ImportError:
        results["torch"] = False
        results["cuda"] = False
        _status("PyTorch", False, "run: pip install torch --index-url https://download.pytorch.org/whl/cu121")

    # flash-attn
    try:
        import flash_attn
        results["flash_attn"] = True
        _status(f"flash-attn {flash_attn.__version__}", True, "3× faster attention on A6000")
    except ImportError:
        results["flash_attn"] = False
        _status("flash-attn", False,
                "STRONGLY recommended: pip install flash-attn --no-build-isolation")

    # Required packages
    print()
    packages = {
        "transformers":    "4.49.0",
        "accelerate":      "0.30.0",
        "qwen_vl_utils":   "0.0.8",
        "datasets":        "2.19.0",
        "opencv-python":   None,   # checked as cv2
        "tqdm":            None,
    }
    import_names = {
        "opencv-python": "cv2",
        "qwen_vl_utils": "qwen_vl_utils",
    }

    all_pkg_ok = True
    for pkg, min_ver in packages.items():
        imp = import_names.get(pkg, pkg.replace("-", "_"))
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "?")
            results[f"pkg_{pkg}"] = True
            _status(f"  {pkg} {ver}", True)
        except ImportError:
            results[f"pkg_{pkg}"] = False
            all_pkg_ok = False
            _status(f"  {pkg}", False, f"pip install {pkg}>={min_ver or '0'}")

    results["packages_ok"] = all_pkg_ok

    # Disk space
    print()
    try:
        import shutil
        free_gb = shutil.disk_usage("/").free / 1e9
        enough = free_gb >= 35
        results["disk_free_gb"] = free_gb
        _status(f"Free disk: {free_gb:.0f} GB", enough,
                "need ~35 GB for both models")
    except Exception:
        results["disk_free_gb"] = 0

    print()
    ready = results.get("cuda") and results.get("packages_ok")
    if ready:
        print("  System is READY for Qwen local inference.")
    else:
        print("  Fix the issues above before running the pipeline.")

    return results


def _status(label: str, ok: bool, note: str = ""):
    symbol = "OK" if ok else "!!"
    note_str = f"  <- {note}" if (not ok and note) else ""
    print(f"  [{symbol}] {label}{note_str}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_model(model_name: str, cache_dir: str = None) -> Path:
    """
    Download a model from HuggingFace Hub to local cache.
    Uses snapshot_download for atomic, resumable download.
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_url
    except ImportError:
        print("  huggingface_hub not installed. Run: pip install huggingface-hub")
        return None

    print(f"\n  Downloading {model_name}...")
    print(f"  (This is a one-time download. Progress is shown below.)")

    kwargs = {"repo_id": model_name, "repo_type": "model"}
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
        return Path(local_path)
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Alternative: huggingface-cli download {model_name}")
        return None


def check_model_cached(model_name: str, cache_dir: str = None) -> bool:
    """Check if model weights are already in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        hf_cache = cache_dir or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        info = scan_cache_dir(hf_cache)
        for repo in info.repos:
            if repo.repo_id == model_name:
                return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# VLM INFERENCE TEST (Qwen2.5-VL)
# ─────────────────────────────────────────────────────────────────────────────

TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

VLM_TEST_PROMPT = """You are helping a blind user navigate.
Describe this image briefly, including any objects and their positions.
Return JSON: {"objects": [...], "spatial": {...}, "summary": "..."}"""


def test_vlm(model_name: str, gpu_id: int = 0, cache_dir: str = None) -> dict:
    """Run a quick VLM inference test to verify the model works correctly."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import urllib.request
    import tempfile

    print(f"\n  Testing {model_name} on GPU {gpu_id}...")

    # Download a small test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        urllib.request.urlretrieve(TEST_IMAGE_URL, tmp_path)
    except Exception:
        # If download fails, create a synthetic test image
        try:
            from PIL import Image
            img = Image.new("RGB", (224, 224), color=(128, 64, 32))
            img.save(tmp_path)
        except Exception:
            print("  Could not create test image — skipping VLM test")
            return {"success": False, "error": "no_test_image"}

    start = time.time()
    try:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": f"cuda:{gpu_id}",
        }
        # Use flash_attention_2 if available
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        model.eval()

        load_time = time.time() - start
        vram_used = torch.cuda.memory_allocated(gpu_id) / 1e9
        print(f"  Model loaded  : {load_time:.1f}s | VRAM used: {vram_used:.1f} GB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_TEST_PROMPT},
                {"type": "image", "image": f"file://{os.path.abspath(tmp_path)}"},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(f"cuda:{gpu_id}")

        inf_start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )
        inf_time = time.time() - inf_start

        trimmed = out[0][inputs.input_ids.shape[1]:]
        response = processor.decode(trimmed, skip_special_tokens=True).strip()

        print(f"  Inference time: {inf_time:.2f}s")
        print(f"  Response preview: {response[:200]}...")

        # Free GPU memory
        del model, processor, inputs
        torch.cuda.empty_cache()

        return {
            "success": True,
            "load_time_sec": round(load_time, 2),
            "inference_time_sec": round(inf_time, 2),
            "vram_used_gb": round(vram_used, 2),
            "response_preview": response[:200],
        }

    except Exception as e:
        print(f"  VLM test FAILED: {e}")
        return {"success": False, "error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM INFERENCE TEST (Qwen2.5 text)
# ─────────────────────────────────────────────────────────────────────────────

LLM_TEST_PROMPT = """Generate 3 BLV navigation question-answer pairs for this scene:
{"objects": ["chair", "table", "door"], "spatial": {"chair": "12 o'clock 1m", "door": "3 o'clock 3m"}}

Return JSON array: [{"question": "...", "answer": "...", "category": "..."}]"""


def test_llm(model_name: str, gpu_id: int = 0, cache_dir: str = None) -> dict:
    """Run a quick LLM inference test."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Testing {model_name} on GPU {gpu_id}...")

    start = time.time()
    try:
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": f"cuda:{gpu_id}",
        }
        try:
            import flash_attn
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()

        load_time = time.time() - start
        vram_used = torch.cuda.memory_allocated(gpu_id) / 1e9
        print(f"  Model loaded  : {load_time:.1f}s | VRAM used: {vram_used:.1f} GB")

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Return valid JSON only."},
            {"role": "user", "content": LLM_TEST_PROMPT},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(f"cuda:{gpu_id}")

        inf_start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        inf_time = time.time() - inf_start

        response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Inference time: {inf_time:.2f}s")
        print(f"  Response preview: {response[:300]}...")

        del model, tokenizer, inputs
        torch.cuda.empty_cache()

        return {
            "success": True,
            "load_time_sec": round(load_time, 2),
            "inference_time_sec": round(inf_time, 2),
            "vram_used_gb": round(vram_used, 2),
            "response_preview": response[:300],
        }

    except Exception as e:
        print(f"  LLM test FAILED: {e}")
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-GPU BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_throughput(
    vlm_name: str,
    llm_name: str,
    n_gpus: int = 8,
    n_videos_sample: int = 10,
) -> dict:
    """
    Estimate full pipeline throughput by timing a small batch on GPU 0,
    then projecting to full distributed run.
    """
    import torch

    print("\n" + "=" * 60)
    print("  THROUGHPUT BENCHMARK")
    print("=" * 60)

    # VLM benchmark (single GPU, then extrapolate)
    print(f"\n  Benchmarking VLM ({vlm_name}) on GPU 0...")
    vlm_result = test_vlm(vlm_name, gpu_id=0)
    if not vlm_result["success"]:
        print("  VLM benchmark failed")
        return {}

    vlm_sec_per_video = vlm_result["inference_time_sec"]
    print(f"  VLM: {vlm_sec_per_video:.2f}s/video on 1 GPU")

    # LLM benchmark
    print(f"\n  Benchmarking LLM ({llm_name}) on GPU 0...")
    llm_result = test_llm(llm_name, gpu_id=0)
    if not llm_result["success"]:
        print("  LLM benchmark failed")
        return {}

    # Step 3 uses 4 LLM calls per video (3 groups + quality check)
    llm_sec_per_video = llm_result["inference_time_sec"] * 4
    print(f"  LLM: {llm_sec_per_video:.2f}s/video on 1 GPU (4 calls/video)")

    # Projections
    n_videos = 11_909  # AVCaps + Charades

    step2_single = n_videos * vlm_sec_per_video / 3600
    step3_single = n_videos * llm_sec_per_video / 3600
    step2_dist   = step2_single / n_gpus
    step3_dist   = step3_single / n_gpus

    print(f"\n  ─── Projected runtime for {n_videos:,} videos ────────────────")
    print(f"  Step 2 (VLM): {step2_single:.1f}h single GPU → {step2_dist:.1f}h on {n_gpus} GPUs")
    print(f"  Step 3 (LLM): {step3_single:.1f}h single GPU → {step3_dist:.1f}h on {n_gpus} GPUs")
    print(f"  Total        : ~{step2_dist + step3_dist:.1f}h wall-clock on {n_gpus}× A6000")

    return {
        "vlm_sec_per_video": vlm_sec_per_video,
        "llm_sec_per_video": llm_sec_per_video,
        "step2_hours_8gpu": round(step2_dist, 1),
        "step3_hours_8gpu": round(step3_dist, 1),
        "total_hours_8gpu": round(step2_dist + step3_dist, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT READY COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def print_ready_commands():
    print("\n" + "=" * 60)
    print("  READY — NEXT STEPS")
    print("=" * 60)
    print("""
  1. Download datasets:
     python download_datasets.py --data_dir /data

  2. Extract frames (motion-based, 24/video, parallel):
     python step1_extract_frames.py \\
         --video_list /data/combined_index.json \\
         --output_dir ./keyframes \\
         --method motion --n_frames 24 --workers 4

  3. Generate scene summaries (distributed 8× A6000):
     python step2_scene_summary.py \\
         --manifest frame_manifest.json \\
         --output scene_summaries.json \\
         --combined_index /data/combined_index.json \\
         --distributed --n_gpus 8

  4. Generate BLV queries (distributed 8× A6000):
     python step3_generate_queries.py \\
         --summaries scene_summaries.json \\
         --output blv_queries.json \\
         --n_pairs 20 \\
         --combined_index /data/combined_index.json \\
         --distributed --n_gpus 8

  5. Convert to training format:
     python step4_convert_format.py \\
         --queries blv_queries.json \\
         --manifest frame_manifest.json \\
         --summaries scene_summaries.json \\
         --output_dir training_data \\
         --formats jsonl smolvla hf --min_quality 2

  OR — run everything with one command:
     python run_pipeline.py \\
         --video_list /data/combined_index.json \\
         --output_root ./pipeline_output \\
         --n_pairs 20 --distributed --n_gpus 8
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Setup and verify Qwen2.5 models for BLV navigation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check_only", action="store_true",
                        help="Check environment without downloading models")
    parser.add_argument("--download_only", action="store_true",
                        help="Download models without running inference tests")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run throughput benchmark after setup")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU to use for single-GPU tests")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--vlm_only", action="store_true", help="Only setup VLM (Step 2)")
    parser.add_argument("--llm_only", action="store_true", help="Only setup LLM (Step 3)")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"HF cache dir: {args.cache_dir}")

    # ── 1. System check ────────────────────────────────────────────────────
    sys_info = check_system()

    if args.check_only:
        print_ready_commands()
        return

    if not sys_info.get("cuda"):
        print("\n  Cannot proceed without CUDA. Install PyTorch with CUDA first:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return

    # ── 2. Download models ─────────────────────────────────────────────────
    to_setup = []
    if not args.llm_only:
        to_setup.append(("vlm", MODELS["vlm"]))
    if not args.vlm_only:
        to_setup.append(("llm", MODELS["llm"]))

    print(f"\n{'='*60}")
    print("  MODEL DOWNLOADS")
    print(f"{'='*60}")

    downloaded = {}
    for key, info in to_setup:
        print(f"\n  {info['role']}")
        print(f"  Model : {info['name']}")
        print(f"  VRAM  : ~{info['vram_gb']} GB per GPU")
        print(f"  Disk  : ~{info['disk_gb']} GB")

        if check_model_cached(info["name"], args.cache_dir):
            print(f"  Already cached. Skipping download.")
            downloaded[key] = True
        else:
            path = download_model(info["name"], args.cache_dir)
            downloaded[key] = path is not None

    if args.download_only:
        print("\n  Download complete.")
        print_ready_commands()
        return

    # ── 3. Inference tests ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INFERENCE TESTS")
    print(f"{'='*60}")

    test_results = {}

    if not args.llm_only and downloaded.get("vlm"):
        test_results["vlm"] = test_vlm(
            MODELS["vlm"]["name"],
            gpu_id=args.gpu_id,
            cache_dir=args.cache_dir,
        )
        if test_results["vlm"]["success"]:
            print(f"  [OK] VLM test passed")
        else:
            print(f"  [!!] VLM test FAILED: {test_results['vlm'].get('error')}")

    if not args.vlm_only and downloaded.get("llm"):
        test_results["llm"] = test_llm(
            MODELS["llm"]["name"],
            gpu_id=args.gpu_id,
            cache_dir=args.cache_dir,
        )
        if test_results["llm"]["success"]:
            print(f"  [OK] LLM test passed")
        else:
            print(f"  [!!] LLM test FAILED: {test_results['llm'].get('error')}")

    # ── 4. Benchmark ───────────────────────────────────────────────────────
    bench_results = {}
    if args.benchmark:
        bench_results = benchmark_throughput(
            vlm_name=MODELS["vlm"]["name"],
            llm_name=MODELS["llm"]["name"],
            n_gpus=args.n_gpus,
        )

    # ── 5. Save results ────────────────────────────────────────────────────
    report = {
        "system": sys_info,
        "models_downloaded": downloaded,
        "inference_tests": test_results,
        "benchmark": bench_results,
    }

    # Filter non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    report_path = Path("qwen_setup_report.json")
    report_path.write_text(json.dumps(make_serializable(report), indent=2))
    print(f"\n  Setup report saved: {report_path}")

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SETUP SUMMARY")
    print(f"{'='*60}")

    vlm_ok = test_results.get("vlm", {}).get("success", not args.llm_only)
    llm_ok = test_results.get("llm", {}).get("success", not args.vlm_only)

    if args.llm_only:
        all_ok = llm_ok
    elif args.vlm_only:
        all_ok = vlm_ok
    else:
        all_ok = vlm_ok and llm_ok

    if not args.llm_only:
        symbol = "[OK]" if vlm_ok else "[!!]"
        print(f"  {symbol} Qwen2.5-VL-7B-Instruct  (Step 2 vision model)")
    if not args.vlm_only:
        symbol = "[OK]" if llm_ok else "[!!]"
        print(f"  {symbol} Qwen2.5-7B-Instruct     (Step 3 text model)")

    if all_ok:
        print("\n  Both models are working. You are ready to run the pipeline.")
    else:
        print("\n  Some tests failed. Check the errors above.")
        print("  Common fixes:")
        print("    - pip install flash-attn --no-build-isolation")
        print("    - pip install transformers --upgrade")
        print("    - pip install qwen-vl-utils --upgrade")

    print_ready_commands()


if __name__ == "__main__":
    main()
