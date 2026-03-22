# BLV Navigation Dataset Pipeline
## AVCaps + Charades → SmolVLA Instruction-Tuning Dataset

---

## Overview

```
video → keyframes → VLM scene summary → BLV queries → SmolVLA training samples
```

**Estimated output**: ~240K–480K samples from 12K videos (Charades 9.8K + AVCaps 2K)

---

## Files

| File | Step | Description |
|---|---|---|
| `step1_extract_frames.py` | 1 | Extract keyframes (uniform / motion / CLIP) |
| `step2_scene_summary.py`  | 2 | Generate structured scene JSON via VLM teacher |
| `step3_generate_queries.py` | 3 | Generate BLV question–answer pairs |
| `step4_convert_format.py` | 4 | Convert to JSONL / HuggingFace / SmolVLA format |
| `run_pipeline.py`         | — | **Orchestrates all steps end-to-end** |

---

## Quick Start

### Install dependencies

```bash
pip install opencv-python tqdm openai
# For HuggingFace dataset output:
pip install datasets pyarrow
# For local Qwen VLM:
pip install transformers accelerate qwen-vl-utils
# For CLIP frame selection:
pip install transformers torch torchvision
```

### Run full pipeline (GPT-4o teacher, cheapest query model)

```bash
python run_pipeline.py \
  --video_dir /data/charades_videos \
  --output_root ./pipeline_output \
  --api_key sk-... \
  --vlm_backend gpt4o \
  --query_backend openai \
  --query_model gpt-4o-mini \
  --n_pairs 20 \
  --formats jsonl smolvla
```

### Estimate API cost first (no execution)

```bash
python run_pipeline.py \
  --video_dir /data/videos \
  --output_root ./out \
  --estimate_only
```

### Fully local (no API costs)

```bash
python run_pipeline.py \
  --video_dir /data/videos \
  --output_root ./out \
  --vlm_backend qwen \
  --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
  --query_backend local \
  --n_pairs 20
```

### Skip already-completed steps

```bash
# Already ran Step 1 and 2, resume from Step 3:
python run_pipeline.py \
  --video_dir /data/videos \
  --output_root ./out \
  --skip_steps 1 2 \
  --api_key sk-...
```

---

## Step-by-step usage

### Step 1: Extract frames

```bash
python step1_extract_frames.py \
  --video_dir /data/videos \
  --output_dir ./keyframes \
  --n_frames 24 \
  --method motion \       # uniform | motion | clip
  --manifest frame_manifest.json
```

### Step 2: Scene summaries

```bash
# GPT-4o (best quality)
python step2_scene_summary.py \
  --manifest frame_manifest.json \
  --output scene_summaries.json \
  --backend gpt4o \
  --api_key $OPENAI_API_KEY

# Local Qwen2.5-VL
python step2_scene_summary.py \
  --manifest frame_manifest.json \
  --output scene_summaries.json \
  --backend qwen \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct
```

### Step 3: Generate BLV queries

```bash
python step3_generate_queries.py \
  --summaries scene_summaries.json \
  --output blv_queries.json \
  --n_pairs 20 \
  --backend openai \
  --openai_model gpt-4o-mini \
  --api_key $OPENAI_API_KEY
```

### Step 4: Convert to training format

```bash
python step4_convert_format.py \
  --queries blv_queries.json \
  --manifest frame_manifest.json \
  --output_dir training_data \
  --formats jsonl smolvla hf
```

---

## Output formats

### JSONL (flat, most compatible)
```
training_data/dataset.jsonl
```
```json
{"video_id": "v_AbCd", "frames": ["keyframes/v_AbCd/frame_000.jpg", ...], "instruction": "Can I walk forward safely?", "response": "No. A chair is blocking your path. Move left.", "category": "obstacle_detection"}
```

### SmolVLA shards
```
training_data/smolvla_shards/smolvla_train_000000.jsonl
training_data/smolvla_shards/smolvla_train_000001.jsonl
...
```

### HuggingFace Dataset
```
training_data/hf_dataset/
  train/
  validation/
```
Load with:
```python
from datasets import load_from_disk
ds = load_from_disk("training_data/hf_dataset")
```

---

## Approximate API costs (12K videos, 8 frames/video, 20 pairs/video)

| Config | Step 2 | Step 3 | Total |
|---|---|---|---|
| GPT-4o + GPT-4o | ~$96 | ~$120 | **~$216** |
| GPT-4o + GPT-4o-mini | ~$96 | ~$12 | **~$108** |
| GPT-4o-mini + GPT-4o-mini | ~$15 | ~$12 | **~$27** |
| Fully local (Qwen) | $0 | $0 | **$0** |

Run `--estimate_only` to get exact estimate for your video count.

---

## QA categories generated

1. `scene_understanding` — "What kind of place is this?"
2. `object_awareness` — "What objects are near me?"
3. `spatial_relationships` — "Where is the door relative to me?"
4. `navigation_guidance` — "How do I reach the exit?"
5. `obstacle_detection` — "Can I walk forward safely?"
6. `human_activity` — "Is anyone nearby?"
7. `audio_awareness` — "What sounds might I hear here?"
