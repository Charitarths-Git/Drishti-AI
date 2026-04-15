"""
evaluate_blv.py
===============
Evaluates SmolVLM outputs using Qwen2.5-7B-Instruct as the GPT OSS judge.

Metrics:
  MCF — Mean Confidence Factor (0–1): completeness + confidence of response
  NAF — Navigation Accuracy Factor (0–1): spatial accuracy, obstacles, directions

Final output: mean MCF, mean NAF, combined score, per-category breakdown.

Usage:
    # Single GPU
    python evaluate_blv.py \
        --predictions ./inference_outputs.json \
        --evaluator   /usershome/cs671_user3/models/Qwen2.5-7B-Instruct \
        --output      ./evaluation_results.json

    # Multi-GPU (shards evaluation across GPUs — faster)
    CUDA_VISIBLE_DEVICES=5,6 python evaluate_blv.py \
        --predictions ./inference_outputs.json \
        --evaluator   /usershome/cs671_user3/models/Qwen2.5-7B-Instruct \
        --output      ./evaluation_results.json \
        --distributed --n_gpus 2
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation prompts
# ─────────────────────────────────────────────────────────────────────────────

MCF_PROMPT = """You are evaluating a navigation AI assistant for blind and low-vision users.

Score the response on CONFIDENCE and COMPLETENESS (MCF).

High MCF (close to 1.0):
- Gives specific, complete information
- Directly answers the question
- No unnecessary hedging

Low MCF (close to 0.0):
- Vague or incomplete answer
- Does not address the question
- Unhelpful for a blind person

Question: {instruction}
Response: {prediction}

Respond ONLY with valid JSON, nothing else:
{{"mcf": <number between 0.0 and 1.0>, "reason": "<one sentence>"}}"""

NAF_PROMPT = """You are evaluating a navigation AI assistant for blind and low-vision users.

Score the response on NAVIGATION ACCURACY (NAF).

High NAF (close to 1.0):
- Uses clock-face directions (12 o'clock, 3 o'clock, etc.)
- Gives metric distances (1 meter, 2 meters, etc.)
- Identifies obstacles clearly with positions
- Gives actionable safe directions
- Practically useful for navigating blind

Low NAF (close to 0.0):
- Vague directions ("nearby", "over there", "to the side")
- No distances mentioned
- Misses obvious obstacles
- Not actionable for real navigation

Question: {instruction}
Response: {prediction}

Respond ONLY with valid JSON, nothing else:
{{"naf": <number between 0.0 and 1.0>, "reason": "<one sentence>"}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Qwen evaluator
# ─────────────────────────────────────────────────────────────────────────────

class QwenEvaluator:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        print(f"[Evaluator GPU {device}] Loading {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
            local_files_only=True,
        )
        self.model.eval()
        self.device = device
        mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"[Evaluator {device}] Loaded. VRAM used: {mem:.1f} GB")

    def _call(self, prompt: str, max_new_tokens: int = 80) -> str:
        msgs = [
            {"role": "system", "content": "You are a precise evaluator. Return only valid JSON with no extra text."},
            {"role": "user",   "content": prompt},
        ]
        text   = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )

        n_in  = inputs["input_ids"].shape[1]
        reply = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return reply

    def _parse(self, text: str, key: str):
        # Try strict JSON extraction first
        try:
            m = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                val = obj.get(key)
                if val is not None:
                    return max(0.0, min(1.0, float(val))), obj.get("reason", "")
        except Exception:
            pass

        # Fallback: regex
        m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m:
            return max(0.0, min(1.0, float(m.group(1)))), ""

        return None, f"parse_failed: {text[:80]}"

    def score(self, instruction: str, prediction: str):
        """Returns (mcf, naf, mcf_reason, naf_reason)."""
        if not prediction.strip():
            return 0.0, 0.0, "empty response", "empty response"

        mcf_reply  = self._call(MCF_PROMPT.format(instruction=instruction, prediction=prediction))
        naf_reply  = self._call(NAF_PROMPT.format(instruction=instruction, prediction=prediction))

        mcf, mcf_r = self._parse(mcf_reply, "mcf")
        naf, naf_r = self._parse(naf_reply, "naf")

        mcf = mcf if mcf is not None else 0.5
        naf = naf if naf is not None else 0.5

        return mcf, naf, mcf_r, naf_r


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def eval_worker(gpu_id: int, samples: list, evaluator_path: str, output_path: str):
    device    = f"cuda:{gpu_id}"
    evaluator = QwenEvaluator(evaluator_path, device=device)

    # Resume from checkpoint
    results  = []
    done_ids = set()
    if Path(output_path).exists():
        try:
            results  = json.load(open(output_path))
            done_ids = {r["video_id"] + r["instruction"] for r in results}
            print(f"[GPU {gpu_id}] Resuming: {len(done_ids)} already scored")
        except Exception:
            pass

    remaining = [s for s in samples if (s["video_id"] + s["instruction"]) not in done_ids]

    for sample in tqdm(remaining, desc=f"Eval GPU {gpu_id}", position=gpu_id):
        mcf, naf, mcf_r, naf_r = evaluator.score(
            sample["instruction"], sample["prediction"]
        )
        results.append({
            "video_id":    sample["video_id"],
            "category":    sample.get("category", ""),
            "instruction": sample["instruction"],
            "reference":   sample.get("reference", ""),
            "prediction":  sample["prediction"],
            "mcf":         mcf,
            "naf":         naf,
            "combined":    (mcf + naf) / 2,
            "mcf_reason":  mcf_r,
            "naf_reason":  naf_r,
        })

        if len(results) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[GPU {gpu_id}] Done → {output_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_report(results: list, output_path: str):
    if not results:
        print("No results.")
        return

    n        = len(results)
    mean_mcf = sum(r["mcf"] for r in results) / n
    mean_naf = sum(r["naf"] for r in results) / n
    combined = (mean_mcf + mean_naf) / 2

    print("\n" + "=" * 65)
    print("  BLV EVALUATION REPORT")
    print("=" * 65)
    print(f"  Samples evaluated     : {n:,}")
    print(f"  MCF (Confidence)      : {mean_mcf:.4f}")
    print(f"  NAF (Navigation)      : {mean_naf:.4f}")
    print(f"  Combined Score        : {combined:.4f}")

    # Per-category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.get("category", "unknown")].append(r)

    print(f"\n  {'Category':<32} {'N':>5}  {'MCF':>6}  {'NAF':>6}  {'Combined':>8}")
    print("  " + "-" * 60)
    cat_summary = {}
    for cat, items in sorted(by_cat.items()):
        cn   = len(items)
        cmcf = sum(i["mcf"] for i in items) / cn
        cnaf = sum(i["naf"] for i in items) / cn
        cc   = (cmcf + cnaf) / 2
        cat_summary[cat] = {"n": cn, "mcf": round(cmcf, 4), "naf": round(cnaf, 4), "combined": round(cc, 4)}
        print(f"  {cat:<32} {cn:>5}  {cmcf:>6.3f}  {cnaf:>6.3f}  {cc:>8.3f}")

    # 3 worst and 3 best
    sorted_r = sorted(results, key=lambda x: x["combined"])
    print(f"\n  3 lowest scoring:")
    for r in sorted_r[:3]:
        print(f"    [{r['combined']:.2f}] Q: {r['instruction'][:55]}")
        print(f"           A: {r['prediction'][:75]}")
    print(f"\n  3 highest scoring:")
    for r in sorted_r[-3:]:
        print(f"    [{r['combined']:.2f}] Q: {r['instruction'][:55]}")
        print(f"           A: {r['prediction'][:75]}")
    print("=" * 65)

    # Save summary JSON
    summary = {
        "n_samples":     n,
        "mean_mcf":      round(mean_mcf, 4),
        "mean_naf":      round(mean_naf, 4),
        "combined":      round(combined, 4),
        "per_category":  cat_summary,
    }
    summary_path = output_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="./inference_outputs.json")
    p.add_argument("--evaluator",   default="/usershome/cs671_user3/models/Qwen2.5-7B-Instruct")
    p.add_argument("--output",      default="./evaluation_results.json")
    p.add_argument("--n_samples",   type=int, default=500)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--n_gpus",      type=int, default=1)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions) as f:
        all_samples = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    samples = all_samples[:args.n_samples]
    print(f"Evaluating {len(samples)} samples")

    if args.distributed and args.n_gpus > 1:
        shards       = [[] for _ in range(args.n_gpus)]
        output_paths = [args.output.replace(".json", f"_gpu{i}.json") for i in range(args.n_gpus)]

        for i, s in enumerate(samples):
            shards[i % args.n_gpus].append(s)

        processes = []
        for gpu_id in range(args.n_gpus):
            p = mp.Process(
                target=eval_worker,
                args=(gpu_id, shards[gpu_id], args.evaluator, output_paths[gpu_id]),
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

    else:
        all_results = eval_worker(0, samples, args.evaluator, args.output)

    print_and_save_report(all_results, args.output)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
