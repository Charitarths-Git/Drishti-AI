"""
analyze_teacher_results.py
==========================
Parses the output of evaluate_blv.py for all teacher models, compares
their MCF/NAF navigation performance, analyzes their response complexity,
and scores their suitability as a distillation teacher for a 500M student model.

Usage:
    python analyze_teacher_results.py \
        --results_dir ./teacher_eval_results \
        --report
"""

import os
import re
import json
import argparse
import statistics
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Teacher Registry
# ─────────────────────────────────────────────────────────────────────────────

TEACHERS = {
    "internvl3": {"name": "InternVL3-78B", "size_b": 78},
    "tarsier2":  {"name": "Tarsier2-Recap-7B", "size_b": 7},
    "qwen72b":   {"name": "Qwen2.5-VL-72B","size_b": 72},
    "qwen7b":    {"name": "Qwen2.5-VL-7B", "size_b": 7},
}

# ─────────────────────────────────────────────────────────────────────────────
# Complexity & Formatting Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_complexity_metrics(predictions: list) -> dict:
    """"Analyze the textual complexity of the generated answers."""
    lengths = [len(p.split()) for p in predictions]
    vocab = set(word.lower() for p in predictions for word in p.split())

    # Count structured formatting (bad for mobile TTS)
    bullet_lists = sum(1 for p in predictions if re.search(r'^\s*[-*]\s', p, re.MULTILINE))
    numbered_lists = sum(1 for p in predictions if re.search(r'^\s*\d+\.\s', p, re.MULTILINE))
    bold_tags = sum(1 for p in predictions if '**' in p)
    
    # Distance/Clock formatting (good for BLV)
    has_clock = sum(1 for p in predictions if re.search(r"\d+\s*o'clock", p, re.IGNORECASE))
    has_meters = sum(1 for p in predictions if re.search(r"\d+\s*(meters?|m\b)", p, re.IGNORECASE))

    n = len(predictions) if predictions else 1

    return {
        "avg_length": statistics.mean(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "unique_vocab_size": len(vocab),
        "pct_bullets": (bullet_lists / n) * 100,
        "pct_numbered": (numbered_lists / n) * 100,
        "pct_bold": (bold_tags / n) * 100,
        "pct_has_clock": (has_clock / n) * 100,
        "pct_has_meters": (has_meters / n) * 100,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_models(results_dir: str, print_report: bool = True):
    dir_path = Path(results_dir)
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} not found.")
        return

    teacher_stats = {}

    for key, info in TEACHERS.items():
        eval_file = dir_path / f"eval_{key}.json"
        if not eval_file.exists():
            continue

        try:
            with open(eval_file, "r") as f:
                data = json.load(f)
                
            if not data:
                continue

            predictions = [d["prediction"] for d in data if "prediction" in d]
            mcf = sum(d["mcf"] for d in data) / len(data)
            naf = sum(d["naf"] for d in data) / len(data)
            combined = (mcf + naf) / 2
            
            complexity = compute_complexity_metrics(predictions)
            
            # Distillation Suitability (0-100)
            # High NAF/MCF is good. Low length/bullets is good for small students. Clock/meters is good.
            base_score = combined * 100
            length_penalty = -abs(30 - complexity["avg_length"]) * 0.5  # Penalize if too short or too long
            format_penalty = -(complexity["pct_bullets"] + complexity["pct_numbered"] + complexity["pct_bold"]) * 0.3
            blv_bonus = (complexity["pct_has_clock"] + complexity["pct_has_meters"]) * 0.2
            
            suitability = max(0, min(100, base_score + length_penalty + format_penalty + blv_bonus))

            teacher_stats[key] = {
                "name": info["name"],
                "mcf": mcf * 100,
                "naf": naf * 100,
                "combined": combined * 100,
                "complexity": complexity,
                "suitability_score": suitability
            }
        except Exception as e:
            print(f"Error parsing {eval_file}: {e}")

    if not teacher_stats:
        print("No evaluation files found. Run evaluate_blv.py first!")
        return {}

    if print_report:
        _print_summary(teacher_stats)

    return teacher_stats

def _print_summary(stats: dict):
    print("\n" + "═" * 80)
    print(f"  BLV TEACHER MODEL ANALYSIS")
    print("═" * 80)
    
    # ─── 1. PERFORMANCE TABLE ────────────────────────────────────────────────
    print("\n  1. Navigation Accuracy (vs Baseline)")
    print("  " + "─" * 70)
    print(f"  {'Model':<25s} | {'MCF':>7s} | {'NAF':>7s} | {'Combined':>8s}")
    print("  " + "─" * 70)
    
    sorted_by_score = sorted(stats.values(), key=lambda x: x["combined"], reverse=True)
    for s in sorted_by_score:
        print(f"  {s['name']:<25s} | {s['mcf']:>6.1f}% | {s['naf']:>6.1f}% | {s['combined']:>7.1f}%")
        
    # ─── 2. RESPONSE STYLE FOR BLV ───────────────────────────────────────────
    print("\n\n  2. Response Style (Suitability for TTS)")
    print("  " + "─" * 78)
    print(f"  {'Model':<25s} | {'Avg Len':>7s} | {'Markdown %':>12s} | {'Clock/Mtrs %':>14s}")
    print("  " + "─" * 78)
    
    for s in sorted_by_score:
        c = s["complexity"]
        md_pct = c["pct_bullets"] + c["pct_numbered"] + c["pct_bold"]
        nav_pct = (c["pct_has_clock"] + c["pct_has_meters"]) / 2
        print(f"  {s['name']:<25s} | {c['avg_length']:>5.0f} w | {md_pct:>11.1f}% | {nav_pct:>13.1f}%")

    # ─── 3. DISTILLATION RECOMMENDATION ──────────────────────────────────────
    print("\n\n  3. Distillation Suitability")
    print("  Teacher for 500M Student — Score balances accuracy, conciseness, and low markdown.")
    print("  " + "─" * 50)
    
    sorted_by_suitability = sorted(stats.values(), key=lambda x: x["suitability_score"], reverse=True)
    for i, s in enumerate(sorted_by_suitability):
        badge = "🏆 BEST" if i == 0 else ""
        print(f"  {s['name']:<25s} | {s['suitability_score']:>5.1f} / 100   {badge}")

    print("\n" + "═" * 80)
    print(f"  Analysis written. Pick the Best Teacher to train SmolVLM!")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=".", help="Directory containing eval_*.json files")
    parser.add_argument("--report", action="store_true", help="Print full analysis report")
    args = parser.parse_args()
    
    analyze_models(args.results_dir, print_report=args.report)
