import os
import json
import argparse
from pathlib import Path

TEACHERS = {
    "internvl3": {"name": "InternVL3-78B", "size_b": 78},
    "tarsier2":  {"name": "Tarsier2-Recap-7B", "size_b": 7},
    "qwen72b":   {"name": "Qwen2.5-VL-72B","size_b": 72},
    "qwen7b":    {"name": "Qwen2.5-VL-7B", "size_b": 7},
}

BENCHMARKS = {
    "egoschema": "EgoSchema",
    "perceptiontest": "PerceptionTest",
    "videomme": "Video-MME"
}

def analyze_lmms(results_dir="."):
    all_results_file = Path(results_dir) / "teacher_eval_all_results.json"
    if not all_results_file.exists():
        print(f"Error: {all_results_file} not found.")
        return
        
    with open(all_results_file) as f:
        data = json.load(f)
        
    print("\n" + "="*80)
    print("  LMMS-EVAL PUBLIC BENCHMARK RESULTS")
    print("="*80)
    
    header = f"  {'Model':<25} | " + " | ".join(f"{BENCHMARKS[t]:<15}" for t in BENCHMARKS)
    print(header)
    print("  " + "-"*75)
    
    for model_key, info in TEACHERS.items():
        row = f"  {info['name']:<25} | "
        
        for task_key in BENCHMARKS:
            run_name = f"{model_key}_{task_key}"
            res = data.get(run_name, {})
            
            score_str = "N/A"
            if res.get("status") in ("success", "cached"):
                results = res.get("results", {})
                for k, v in results.items():
                    if isinstance(v, dict):
                        for m in ["accuracy", "acc", "score", "exact_match"]:
                            if m in v:
                                val = v[m]
                                score = val * 100 if val <= 1.0 else val
                                score_str = f"{score:.1f}%"
                                break
                    else:
                        for m in ["accuracy", "acc", "score", "exact_match"]:
                            if m in results:
                                val = results[m]
                                score = val * 100 if val <= 1.0 else val
                                score_str = f"{score:.1f}%"
                                break
            row += f"{score_str:<15} | "
            
        print(row.strip(" |"))
        
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=".")
    args = parser.parse_args()
    analyze_lmms(args.results_dir)
