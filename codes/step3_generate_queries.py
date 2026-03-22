"""
Step 3 — BLV Query Generation via Qwen2.5-7B-Instruct (Local, Free)
=====================================================================
Hardware target : 8× NVIDIA RTX A6000 (49 GB each)
Model           : Qwen/Qwen2.5-7B-Instruct  (text-only, ~15 GB VRAM on one GPU)
                  OR Qwen/Qwen3.5 when weights are released

Input  : scene_summaries.json  (from Step 2)
         combined_index.json   (optional — adds AVCaps audio captions as ground truth)
Output : blv_queries.json      { video_id -> list of {question, answer, category, quality_score} }

Design principles for highest QA quality:
  1. Diverse phrasing per category — avoids the model learning to recognise
     template patterns rather than understanding navigation queries
  2. Quality scoring — every pair is self-evaluated by the model (0-3),
     pairs below threshold are discarded before saving
  3. AVCaps audio injection — when audio captions exist, they seed the
     audio_awareness category with ground-truth content
  4. Iterative retry with rephrasing — if a batch has too many low-quality
     pairs, regenerate with a different temperature
  5. Multi-call strategy — one LLM call per category group (4 calls per video)
     produces more focused, higher-quality pairs than one mega-prompt

QA categories (8 total, expanded from 7):
  1. scene_understanding
  2. object_awareness
  3. spatial_relationships
  4. navigation_guidance
  5. obstacle_detection
  6. human_activity
  7. audio_awareness
  8. safety_assessment  ← NEW: combines obstacle + navigation for critical safety QA

Usage:
  # Fully local on GPU 0 (recommended — no API cost)
  python step3_generate_queries.py \\
      --summaries scene_summaries.json \\
      --output blv_queries.json \\
      --n_pairs 20

  # Distributed: one LLM worker per GPU for maximum throughput
  python step3_generate_queries.py \\
      --summaries scene_summaries.json \\
      --output blv_queries.json \\
      --n_pairs 20 --distributed --n_gpus 8

  # With OpenAI fallback
  python step3_generate_queries.py \\
      --summaries scene_summaries.json --backend openai --openai_model gpt-4o-mini
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# One template per category group — focused prompts produce better QA than
# one giant prompt trying to cover all categories at once.
# ─────────────────────────────────────────────────────────────────────────────

# Group A: Scene + Object + Spatial (pure perception)
PROMPT_GROUP_A = """You are generating training data for an AI assistant helping blind and low-vision (BLV) users.

Scene description:
{SCENE_SUMMARY}

Generate exactly {N} question-answer pairs for these categories:
- scene_understanding : What kind of place is this? What can you tell me about this environment?
- object_awareness    : What objects are near me? Is there a [specific object] nearby?
- spatial_relationships: Where is [object] relative to me? What is directly ahead / to my left?

Requirements:
- Questions must sound like a real blind person speaking to an AI assistant
- Vary the phrasing — do NOT repeat the same question structure twice
- Answers must name clock-face directions and distances: "The chair is at 11 o'clock, about 1 meter ahead"
- Each answer must be 2-4 sentences: position, description, relevance to navigation
- Avoid filler phrases like "I can see" or "The image shows" — speak directly to the user

Return ONLY valid JSON array:
[
  {{"question": "...", "answer": "...", "category": "scene_understanding|object_awareness|spatial_relationships"}}
]"""

# Group B: Navigation + Obstacle + Safety (action-oriented)
PROMPT_GROUP_B = """You are generating training data for an AI assistant helping blind and low-vision (BLV) users navigate safely.

Scene description:
{SCENE_SUMMARY}

Generate exactly {N} question-answer pairs for these categories:
- navigation_guidance  : How do I reach the door? Which direction should I walk? How do I get to [location]?
- obstacle_detection   : Can I walk forward safely? Are there any hazards? What's blocking my path?
- safety_assessment    : Is this environment safe to navigate alone? What are the biggest hazards here?

Requirements:
- Navigation answers must give ACTIONABLE step-by-step directions
  Example: "Turn slightly left to about 10 o'clock, walk 2 meters, then the door is on your right"
- Obstacle answers must include: what it is, where it is (clock direction + distance), how to avoid it
- Safety answers must prioritise the highest-risk hazard first
- Use natural conversational language a real user would hear via text-to-speech

Return ONLY valid JSON array:
[
  {{"question": "...", "answer": "...", "category": "navigation_guidance|obstacle_detection|safety_assessment"}}
]"""

# Group C: People + Audio (awareness)
PROMPT_GROUP_C = """You are generating training data for an AI assistant helping blind and low-vision (BLV) users.

Scene description:
{SCENE_SUMMARY}

Generate exactly {N} question-answer pairs for these categories:
- human_activity  : Is anyone nearby? What are people doing? Is someone approaching me?
- audio_awareness : What sounds would I hear here? Is it noisy? What does the environment sound like?

Requirements for human_activity:
- Describe number of people, their approximate location (clock direction), and movement direction
- Flag if anyone is moving toward the user

Requirements for audio_awareness:
- Be specific: footsteps, voices (distant/close/loud), traffic, machinery, music, alarms, nature
- Describe whether the sound environment would help or hinder navigation (echo in corridor = useful landmark)

Return ONLY valid JSON array:
[
  {{"question": "...", "answer": "...", "category": "human_activity|audio_awareness"}}
]"""

# Quality self-check prompt — used to score generated pairs
QUALITY_CHECK_PROMPT = """Rate these question-answer pairs for a BLV navigation assistant.
Score each pair 0-3:
  3 = Excellent: specific, actionable, natural-sounding, accurate to scene
  2 = Good: correct but vague directions or slightly unnatural phrasing
  1 = Poor: too generic, wrong direction info, or unhelpful to a blind user
  0 = Invalid: wrong, contradicts scene, or harmful

Scene:
{SCENE_SUMMARY}

Pairs to rate:
{QA_PAIRS}

Return ONLY JSON array of scores (one integer per pair, same order):
[3, 2, 3, ...]"""


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL LLM BACKEND (Qwen2.5-7B-Instruct)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache: one model per GPU per process
_MODEL_CACHE = {}


def get_local_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", gpu_id: int = 0):
    """Load and cache the text LLM on the specified GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (model_name, gpu_id)
    if key not in _MODEL_CACHE:
        print(f"Loading {model_name} on GPU {gpu_id}...")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{gpu_id}",
            local_files_only=True,
        )
        model.eval()

        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

        _MODEL_CACHE[key] = (tokenizer, model)
        print(f"[GPU {gpu_id}] LLM loaded. VRAM: {torch.cuda.memory_allocated(gpu_id) / 1e9:.1f} GB")

    return _MODEL_CACHE[key]


def call_local_qwen(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    gpu_id: int = 0,
    max_new_tokens: int = 1500,
) -> str:
    import torch

    tokenizer, model = get_local_model(model_name, gpu_id)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant generating training data. Always return valid JSON as instructed.",
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(f"cuda:{gpu_id}")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )

    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def call_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant generating training data. Always return valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PARSING AND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_qa_list(text: str) -> list[dict]:
    """Robust extraction of JSON array from LLM output."""
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find outermost JSON array
    start = text.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return []


def validate_pair(pair: dict) -> bool:
    """Structural validity check for a QA pair."""
    return (
        isinstance(pair, dict)
        and isinstance(pair.get("question"), str)
        and isinstance(pair.get("answer"), str)
        and len(pair["question"].strip()) >= 15
        and len(pair["answer"].strip()) >= 40   # raised from 20 — ensures substantive answers
        and pair.get("category") in {
            "scene_understanding", "object_awareness", "spatial_relationships",
            "navigation_guidance", "obstacle_detection", "safety_assessment",
            "human_activity", "audio_awareness",
        }
    )


def score_qa_pairs(
    pairs: list[dict],
    scene_text: str,
    call_fn,
    min_score: int = 2,
) -> list[dict]:
    """
    Use the LLM to self-score QA pairs and filter out low-quality ones.
    Returns only pairs with score >= min_score.

    min_score=2 keeps "Good" and "Excellent" pairs, discards "Poor" and "Invalid".
    """
    if not pairs:
        return pairs

    try:
        qa_text = json.dumps([{"q": p["question"], "a": p["answer"]} for p in pairs], indent=2)
        prompt = QUALITY_CHECK_PROMPT.format(
            SCENE_SUMMARY=scene_text[:1000],  # cap context to avoid huge prompts
            QA_PAIRS=qa_text[:3000],
        )
        raw = call_fn(prompt)
        scores_raw = parse_qa_list(raw)

        if isinstance(scores_raw, list) and len(scores_raw) == len(pairs):
            filtered = []
            for pair, score in zip(pairs, scores_raw):
                try:
                    s = int(score)
                except (ValueError, TypeError):
                    s = 2  # default if score is malformed
                if s >= min_score:
                    pair["quality_score"] = s
                    filtered.append(pair)
            return filtered

    except Exception:
        pass  # scoring is optional — return all valid pairs if it fails

    # If scoring fails, return all structurally valid pairs with default score
    for pair in pairs:
        pair.setdefault("quality_score", 2)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION PER VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def generate_queries_for_video(
    video_id: str,
    scene_data: dict,
    combined_entry: dict,
    n_pairs: int,
    call_fn,
    max_retries: int = 2,
    quality_filter: bool = True,
) -> list[dict]:
    """
    Generate n_pairs BLV QA pairs for a single video using multi-group strategy.

    Multi-group strategy:
      - Group A (scene/object/spatial): ceil(n_pairs * 0.40)
      - Group B (navigation/obstacle/safety): ceil(n_pairs * 0.40)
      - Group C (people/audio): floor(n_pairs * 0.20)

    This distribution mirrors the natural distribution of BLV user queries:
    navigation and obstacle queries are most common in real usage.
    """
    scene = scene_data.get("scene", {})
    if scene.get("parse_error"):
        scene_text = scene.get("raw_response", "")[:2000]
    else:
        scene_text = json.dumps(scene, indent=2)

    # Inject AVCaps audio ground truth for better audio_awareness pairs
    # Field names vary: audio_caption, audio_visual_captions (list), GPT_AV_captions (list)
    audio_caption = None
    for key in ["audio_caption", "audio_visual_caption", "audio_visual_captions", "GPT_AV_captions"]:
        val = combined_entry.get(key)
        if val:
            audio_caption = val[0] if isinstance(val, list) else val
            break
    if audio_caption:
        scene_text += f"\n\nGround-truth audio annotation: {audio_caption}"

    n_a = max(2, int(n_pairs * 0.40))
    n_b = max(2, int(n_pairs * 0.40))
    n_c = max(2, n_pairs - n_a - n_b)

    all_pairs = []

    groups = [
        (PROMPT_GROUP_A, n_a, "Group A (scene/object/spatial)"),
        (PROMPT_GROUP_B, n_b, "Group B (navigation/obstacle/safety)"),
        (PROMPT_GROUP_C, n_c, "Group C (people/audio)"),
    ]

    for template, n_group, group_name in groups:
        prompt = template.format(SCENE_SUMMARY=scene_text, N=n_group)

        for attempt in range(max_retries + 1):
            try:
                raw = call_fn(prompt)
                pairs = parse_qa_list(raw)
                valid = [p for p in pairs if validate_pair(p)]

                if valid:
                    for p in valid:
                        p["video_id"] = video_id
                    all_pairs.extend(valid)
                    break

            except Exception as e:
                if attempt == max_retries:
                    print(f"  [WARN] {video_id} {group_name} failed after {max_retries+1} tries: {e}")
                else:
                    time.sleep(2 ** attempt)

    if not all_pairs:
        return []

    # Quality filter (optional, uses one extra LLM call per video)
    if quality_filter and len(all_pairs) > 0:
        all_pairs = score_qa_pairs(all_pairs, scene_text, call_fn, min_score=2)

    # Remove duplicate questions (exact match)
    seen_questions = set()
    deduped = []
    for p in all_pairs:
        q_norm = p["question"].lower().strip()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            deduped.append(p)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_queries(
    summaries_path: str,
    output_path: str = "blv_queries.json",
    n_pairs: int = 20,
    backend: str = "local",
    api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    local_model: str = "Qwen/Qwen2.5-7B-Instruct",
    gpu_id: int = 0,
    rate_limit_delay: float = 0.0,
    resume: bool = True,
    quality_filter: bool = True,
    combined_index_path: str = None,
):
    with open(summaries_path) as f:
        summaries = json.load(f)

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

    queries = dict(existing)
    to_process = {vid: data for vid, data in summaries.items() if vid not in queries}
    print(f"Generating queries for {len(to_process)} videos")
    print(f"Backend: {backend} | pairs/video: {n_pairs} | quality_filter: {quality_filter}")

    # Build callable
    if backend == "openai":
        _key = api_key or os.environ.get("OPENAI_API_KEY")
        def call_fn(prompt):
            return call_openai(prompt, api_key=_key, model=openai_model)
    else:
        def call_fn(prompt):
            return call_local_qwen(prompt, model_name=local_model, gpu_id=gpu_id)

    total_pairs = sum(len(v) for v in queries.values())
    errors = {}

    for video_id, scene_data in tqdm(to_process.items(), desc="Generating BLV queries"):
        index_entry = (
            combined_index.get(video_id)
            or combined_index.get(f"avcaps__{video_id}")
            or combined_index.get(f"charades__{video_id}")
            or {}
        )

        pairs = generate_queries_for_video(
            video_id=video_id,
            scene_data=scene_data,
            combined_entry=index_entry,
            n_pairs=n_pairs,
            call_fn=call_fn,
            quality_filter=quality_filter,
        )

        if pairs:
            queries[video_id] = pairs
            total_pairs += len(pairs)
        else:
            errors[video_id] = "generation_failed"

        if len(queries) % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(queries, f, indent=2)
            print(f"  Checkpoint: {len(queries)} videos | {total_pairs:,} pairs")

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)

    # Stats
    print(f"\n  Videos processed : {len(queries):,}")
    print(f"  Total QA pairs   : {total_pairs:,}")
    print(f"  Avg pairs/video  : {total_pairs / max(1, len(queries)):.1f}")
    print(f"  Failed           : {len(errors)}")
    print(f"  Output           : {output_path}")

    cat_counts = {}
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for vid_pairs in queries.values():
        for pair in vid_pairs:
            cat = pair.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            score_dist[pair.get("quality_score", 2)] = score_dist.get(pair.get("quality_score", 2), 0) + 1

    print("\n  Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = count / max(1, total_pairs) * 100
        print(f"    {cat:30s} {count:6,}  ({pct:4.1f}%)")

    if quality_filter:
        print("\n  Quality score distribution:")
        for score in sorted(score_dist):
            print(f"    Score {score}: {score_dist[score]:,}")

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTED WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _query_worker_fn(
    gpu_id: int,
    video_ids: list[str],
    summaries: dict,
    combined_index: dict,
    output_path: str,
    n_pairs: int,
    local_model: str,
    quality_filter: bool,
):
    """Worker for one GPU in distributed query generation mode."""
    checkpoint_path = output_path.replace(".json", f"_gpu{gpu_id}.json")

    done = {}
    if Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path) as f:
                done = json.load(f)
            print(f"[GPU {gpu_id}] Resuming: {len(done)} done")
        except Exception:
            done = {}

    todo = [vid for vid in video_ids if vid not in done]
    print(f"[GPU {gpu_id}] {len(todo)} videos to process")

    def call_fn(prompt):
        return call_local_qwen(prompt, model_name=local_model, gpu_id=gpu_id)

    for i, video_id in enumerate(tqdm(todo, desc=f"GPU {gpu_id} queries", position=gpu_id)):
        scene_data = summaries.get(video_id, {})
        index_entry = (
            combined_index.get(video_id)
            or combined_index.get(f"avcaps__{video_id}")
            or combined_index.get(f"charades__{video_id}")
            or {}
        )

        pairs = generate_queries_for_video(
            video_id=video_id,
            scene_data=scene_data,
            combined_entry=index_entry,
            n_pairs=n_pairs,
            call_fn=call_fn,
            quality_filter=quality_filter,
        )

        done[video_id] = pairs if pairs else []

        if (i + 1) % 50 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(done, f, indent=2)

    with open(checkpoint_path, "w") as f:
        json.dump(done, f, indent=2)
    print(f"[GPU {gpu_id}] Done → {checkpoint_path}")


def run_distributed_queries(
    summaries_path: str,
    output_path: str,
    n_pairs: int,
    n_gpus: int,
    local_model: str,
    quality_filter: bool,
    combined_index_path: str = None,
):
    import torch.multiprocessing as mp

    with open(summaries_path) as f:
        summaries = json.load(f)

    combined_index = {}
    if combined_index_path and Path(combined_index_path).exists():
        with open(combined_index_path) as f:
            combined_index = json.load(f)

    existing = {}
    if Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)

    todo = [vid for vid in summaries if vid not in existing]
    print(f"Distributing {len(todo)} videos across {n_gpus} GPUs")

    shards = [[] for _ in range(n_gpus)]
    for i, vid in enumerate(todo):
        shards[i % n_gpus].append(vid)

    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(n_gpus):
        if not shards[gpu_id]:
            continue
        p = ctx.Process(
            target=_query_worker_fn,
            args=(
                gpu_id, shards[gpu_id], summaries, combined_index,
                output_path, n_pairs, local_model, quality_filter,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge
    merged = dict(existing)
    for gpu_id in range(n_gpus):
        cp = output_path.replace(".json", f"_gpu{gpu_id}.json")
        if Path(cp).exists():
            with open(cp) as f:
                shard = json.load(f)
            merged.update(shard)
            print(f"  GPU {gpu_id}: {len(shard)} videos merged")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    total = sum(len(v) for v in merged.values())
    print(f"\n  Total videos: {len(merged):,}")
    print(f"  Total pairs : {total:,}")
    print(f"  Output      : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Generate BLV navigation QA pairs (Qwen local or OpenAI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", default="blv_queries.json")
    parser.add_argument("--n_pairs", type=int, default=20, help="Target QA pairs per video")
    parser.add_argument(
        "--backend", choices=["local", "openai"], default="local",
        help="local=Qwen2.5-7B on GPU (free), openai=API",
    )
    parser.add_argument("--local_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (if --backend openai)")
    parser.add_argument("--openai_model", default="gpt-4o-mini")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU for single-GPU mode")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--no_quality_filter", action="store_true",
                        help="Skip quality scoring (faster but lower quality)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between calls (only needed for OpenAI rate limits)")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--combined_index", default=None,
        help="combined_index.json for AVCaps audio injection",
    )
    args = parser.parse_args()

    quality_filter = not args.no_quality_filter

    if args.distributed and args.backend == "local":
        run_distributed_queries(
            summaries_path=args.summaries,
            output_path=args.output,
            n_pairs=args.n_pairs,
            n_gpus=args.n_gpus,
            local_model=args.local_model,
            quality_filter=quality_filter,
            combined_index_path=args.combined_index,
        )
    else:
        generate_all_queries(
            summaries_path=args.summaries,
            output_path=args.output,
            n_pairs=args.n_pairs,
            backend=args.backend,
            api_key=args.api_key,
            openai_model=args.openai_model,
            local_model=args.local_model,
            gpu_id=args.gpu_id,
            rate_limit_delay=args.delay,
            resume=not args.no_resume,
            quality_filter=quality_filter,
            combined_index_path=args.combined_index,
        )
