# DrishtiAI

**AI-powered, on-device navigation assistant for blind and low-vision (BLV) users.**

DrishtiAI is an Android vision-language assistant that goes beyond generic image captioning to give blind and low-vision users **actionable, movement-oriented guidance** — what's in front of them, where it is, and what to do next — entirely on-device, with no cloud dependency.

> _"Low crate ahead slightly left; open floor to the right; veer right slowly."_

This repository contains the full project: the dataset-generation pipeline, the model fine-tuning and evaluation workflow, and the Android application source.

---

## Table of Contents

- [Overview](#overview)
- [Why On-Device, Why Not Just Captioning](#why-on-device-why-not-just-captioning)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [The Three Project Tracks](#the-three-project-tracks)
  - [1. Dataset Pipeline](#1-dataset-pipeline)
  - [2. Model Fine-Tuning & Evaluation](#2-model-fine-tuning--evaluation)
  - [3. Android Application](#3-android-application)
- [Results](#results)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Tech Stack](#tech-stack)
- [Reports & Further Reading](#reports--further-reading)
- [Team & Acknowledgements](#team--acknowledgements)
- [License](#license)

---

## Overview

Most vision-language assistants describe a scene ("a room with a chair and a table"). That isn't enough for someone navigating that room without sight. DrishtiAI is built around a different question: **what does this person need to know right now to move safely?**

The project has three connected outcomes:

| #   | Outcome                                                         | Result                                                        |
| --- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | A BLV-specific instruction-tuning dataset built from real video | **210,274 samples** generated from Charades + AVCaps          |
| 2   | A compact vision-language model fine-tuned for navigation       | SmolVLM2-500M + LoRA, on-device deployable                    |
| 3   | A working Android app with live navigation guidance             | **~3× latency reduction** (30–35s → 9–11s per guidance cycle) |

---

## Why On-Device, Why Not Just Captioning

Cloud-hosted VLMs are powerful, but for a safety-critical, always-on assistant they introduce three problems:

1. **Latency** — network round-trips make guidance stale by the time it reaches the user.
2. **Privacy** — a camera pointed outward from a person's body captures a continuous stream of their environment, including other people and private indoor spaces.
3. **Reliability** — navigation assistance has to keep working without an internet connection.

DrishtiAI is therefore built as a **fully offline, on-device pipeline**: small/efficient models, navigation-specific instruction tuning, smart frame selection, quantization, and native (C++/JNI) mobile inference — all running locally on a mid-range Android SoC.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   DrishtiAI Android App                   │
│                                                            │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   │
│  │  Chat /     │   │  Video QA    │   │     Live      │   │
│  │  Describe   │   │    Mode      │   │  Navigation   │   │
│  └──────┬──────┘   └──────┬───────┘   └──────┬────────┘   │
│         │                 │                  │            │
│         └─────────────────┴──────────────────┘            │
│                            │                               │
│               ┌────────────▼─────────────┐                 │
│               │   SmolVLM2 Inference      │                 │
│               │   Pipeline (ONNX/GGUF)    │                 │
│               │  ┌─────────────────────┐  │                 │
│               │  │ Vision Encoder      │  │                 │
│               │  │ (SigLIP, frozen ❄️) │  │                 │
│               │  └──────────┬──────────┘  │                 │
│               │  ┌──────────▼──────────┐  │                 │
│               │  │ LM Backbone         │  │                 │
│               │  │ (LoRA fine-tuned)   │  │                 │
│               │  └──────────┬──────────┘  │                 │
│               └─────────────┼─────────────┘                 │
│                              │                               │
│               ┌──────────────▼──────────────┐                │
│               │  NNAPI Delegate (NPU)       │                │
│               └──────────────────────────────┘                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Android TTS Engine                      │    │
│  │      Spatial guidance → voice output                 │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Data journey, camera to spoken guidance (Live Navigation mode):**

```
Camera (CameraX, 1280×720) → 4 frames sampled every 150ms
   → center-crop + resize each to 192×192
   → composite into a 2×2 grid (384×384, matches SigLIP's native input)
   → SigLIP vision encoder → 729 visual tokens (27×27 patches, 1152-dim each)
   → SmolVLM2 LM backbone (LoRA fine-tuned) → one actionable sentence
   → Android TTS → spoken guidance with clock-direction encoding
```

Frames captured while a previous inference is still running are **discarded**, so the model always reasons over the freshest available context rather than a stale queue.

---

## Repository Structure

```
DL_Group-4_BLV/
├── codes/                                    # Core pipeline, training, eval, and reports
│   ├── README.md                             # Dataset pipeline quick-start (steps 1–4)
│   ├── DrishtiAI_Progress_Report.md           # Engineering/perf report (Android optimization)
│   ├── DrishtiAI_Final_Report_and_Presentation_Script.md  # Full technical report + slides
│   ├── DrishtiAI_Technical_Report.tex
│   ├── DrishtiAI/                             # Android app source (Kotlin + C++/JNI)
│   ├── step1_extract_frames.py                # Keyframe extraction (uniform/motion/CLIP)
│   ├── step2_scene_summary.py                 # VLM-generated structured scene JSON
│   ├── step3_generate_queries.py              # BLV-style Q&A pair generation
│   ├── step4_convert_format.py                # JSONL / HuggingFace / SmolVLA conversion
│   ├── run_pipeline.py                        # End-to-end pipeline orchestrator
│   ├── download_datasets.py                   # Charades + AVCaps downloader/indexer
│   ├── download_teacher_models.py              # Teacher VLM downloader (for distillation)
│   ├── setup_qwen.py                           # Local Qwen2.5-VL / Qwen2.5 LLM setup
│   ├── train_smolvla.py                        # LoRA fine-tuning of SmolVLM/SmolVLA
│   ├── inference_smolvla.py                    # Server- and mobile-style inference runner
│   ├── evaluate_blv.py                         # MCF / NAF navigation-specific evaluation
│   └── requirements*.txt                       # Pipeline / training / eval dependencies
├── custom_blv_eval/                           # Teacher-model BLV-suitability analysis
│   ├── analyze_blv_teachers.py
│   ├── eval_egoschema_custom.py
│   └── inference_teachers.py
├── lmms_eval_benchmarks/                      # Public video-QA benchmark harness
│   ├── download_egoschema.py
│   ├── run_teacher_eval.py
│   ├── analyze_public_benchmarks.py
│   └── setup_lmms_eval.sh
├── smolvla_shards/                            # Pre-built SmolVLA training shards (JSONL)
└── codes.zip                                  # Archived snapshot of /codes (+ full dataset.jsonl)
```

> **Note:** the `DrishtiAI/` Android app folder contains the Kotlin/C++ source described throughout this README. Large vendored components (a bundled `llama.cpp`, Gradle build outputs, ONNX/GGUF model weights) are intentionally excluded from version control — see [Getting Started](#getting-started) for how to obtain them.

---

## The Three Project Tracks

### 1. Dataset Pipeline

**Goal:** turn raw video into BLV-style instruction-tuning data.

```
video → keyframes → VLM scene summary → BLV Q&A pairs → training-ready format
```

| Step | Script                      | What it does                                                                                  |
| ---- | --------------------------- | --------------------------------------------------------------------------------------------- |
| 1    | `step1_extract_frames.py`   | Extracts keyframes via uniform, motion-based, or CLIP-based sampling                          |
| 2    | `step2_scene_summary.py`    | Produces a structured scene-description JSON using a VLM teacher (GPT-4o or local Qwen2.5-VL) |
| 3    | `step3_generate_queries.py` | Generates BLV question–answer pairs across 7 categories                                       |
| 4    | `step4_convert_format.py`   | Converts to JSONL, HuggingFace `datasets`, or SmolVLA shard format                            |
| —    | `run_pipeline.py`           | Orchestrates steps 1–4 end-to-end, with cost estimation and step-skipping support             |

**Source video datasets:**

| Dataset      | Role                                                                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Charades** | Indoor activity videos with human-object interactions, kept at original resolution (low-res video loses safety-critical detail like step edges, door handles, small obstacles) |
| **AVCaps**   | Audio-visual captions, used for sound/scene awareness                                                                                                                          |

**QA categories generated:**

1. `scene_understanding` — _"What kind of place is this?"_
2. `object_awareness` — _"What objects are near me?"_
3. `spatial_relationships` — _"Where is the door relative to me?"_
4. `navigation_guidance` — _"How do I reach the exit?"_
5. `obstacle_detection` — _"Can I walk forward safely?"_
6. `human_activity` — _"Is anyone nearby?"_
7. `audio_awareness` — _"What sounds might I hear here?"_

**Approximate generation cost** (12K videos, 8 frames/video, 20 pairs/video):

| Config                              |  Cost |
| ----------------------------------- | ----: |
| GPT-4o (summary) + GPT-4o (queries) | ~$216 |
| GPT-4o + GPT-4o-mini                | ~$108 |
| GPT-4o-mini + GPT-4o-mini           |  ~$27 |
| Fully local (Qwen)                  |    $0 |

Run `python run_pipeline.py --estimate_only` to get an exact estimate for your own video set.

---

### 2. Model Fine-Tuning & Evaluation

| Component              | Detail                                                                                                                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base model             | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`                                                                                                                                                       |
| Fine-tuning method     | LoRA (rank-16), applied **only** to the LM backbone                                                                                                                                                |
| Vision encoder         | SigLIP — **frozen**, zero training cost                                                                                                                                                            |
| Why freeze the encoder | SigLIP already produces rich spatial embeddings from large-scale pretraining; LoRA on the LM head alone was sufficient to steer directional/spatial outputs while avoiding catastrophic forgetting |
| Quantization           | INT4 / FP16 mixed precision                                                                                                                                                                        |
| Inference runtime      | ONNX Runtime Mobile + Android NNAPI delegate (NPU offload)                                                                                                                                         |

**Evaluation** is performed with `evaluate_blv.py`, using two navigation-specific metrics rather than generic caption quality:

- **MCF (Mean Confidence Factor)** — completeness, directness, low hedging
- **NAF (Navigation Accuracy Factor)** — correctness of clock directions, distances, obstacles, and actionable movement

A real production bug was diagnosed and fixed during this phase: the model defaulted to **"6 o'clock" in ~70% of outputs**, regardless of actual obstacle position. This was traced to a directional imbalance in the training data and corrected via dataset rebalancing plus a post-processing confidence-redistribution layer.

---

### 3. Android Application

The `DrishtiAI/` app implements three user-facing modes:

| Feature                                           | Key files                                                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Chat / model management                           | `ChatActivity.kt`, `ChatScreenViewModel.kt`, `ModelsRepository.kt`                  |
| Video QA (record a clip, ask a question)          | `VideoCaptionActivity.kt`, `VideoVisualInputPreparer.kt`                            |
| Live Navigation (hands-free, continuous guidance) | `LiveNavigationActivity.kt`, `VisionSupport.kt`                                     |
| Frame selection                                   | `VideoFrameExtractor.kt`, `QFrameExtractor.kt`                                      |
| Text-to-speech / voice                            | `VoiceController.kt`, `VoiceTextSupport.kt`                                         |
| Native inference (Kotlin → JNI → C++)             | `SmolLM.kt`, `SmolLMManager.kt`, `smollm.cpp`, `LLMInference.cpp` (via `llama.cpp`) |

**Live Navigation loop:**

1. Start camera via CameraX
2. Collect frames for a ~1s window, sampling every 150ms
3. Keep a small candidate pool (max 6 frames)
4. Select 4 frames using MobileNet diversity sampling
5. Compose a 2×2 chronological grid (384×384)
6. Run inference through `SmolLMManager` (context size 2048, max 44 output tokens, temperature 0.45)
7. Stream partial output to the UI; stop early once a complete actionable sentence appears
8. Speak the result via Android TTS, encoded with clock-face directions
9. Open a new collection window

**Frame selection strategies available** (`VideoFrameExtractor.kt`):

| Mode      | Description                            | Best for                                              |
| --------- | -------------------------------------- | ----------------------------------------------------- |
| KEYFRAME  | Fast sync-frame sampling               | General use                                           |
| LUV       | Luminance/color diversity, lightweight | High-contrast scenes                                  |
| MOBILENET | MobileNetV3 feature diversity          | **Default for Live Navigation** — best mobile balance |
| UNIFORM   | Evenly spaced frames                   | General scenes                                        |
| QFRAME    | Quality-ranked CLIP/ONNX selection     | Blurry/motion-heavy footage                           |

---

## Results

### Final dataset

| Metric                        |                                  Value |
| ----------------------------- | -------------------------------------: |
| Total samples                 |                                210,274 |
| Charades-sourced samples      |                                147,014 |
| AVCaps-sourced samples        |                                 37,682 |
| Largest category              | Spatial relationships — 40,682 samples |
| Navigation + obstacle samples |                                 69,231 |

### Evaluation (MCF / NAF, by frame-selection strategy)

| Frame Mode                      |   MCF |   NAF |  Combined | Notes                            |
| ------------------------------- | ----: | ----: | --------: | -------------------------------- |
| LUV (mobile baseline)           | 0.592 | 0.363 |     0.478 | Fast, weaker navigation accuracy |
| QFrame                          | 0.627 | 0.757 |     0.692 | Quality-biased, loses diversity  |
| **MobileNet (mobile, default)** | 0.773 | 0.714 | **0.743** | Best mobile balance              |
| CLIP (server baseline)          | 0.736 | 0.747 |     0.741 | Best server-side quality         |

### Category-wise results (CLIP/server baseline)

| Category              | Combined Score |
| --------------------- | -------------: |
| Object awareness      |          0.837 |
| Spatial relationships |          0.790 |
| Human activity        |          0.775 |
| Obstacle detection    |          0.762 |
| Safety assessment     |          0.725 |
| Navigation guidance   |          0.691 |
| Scene understanding   |          0.689 |
| Audio awareness       |          0.604 |

> Public benchmark numbers (EgoSchema, PerceptionTest, Video-MME) for teacher-model comparison are documented in the [Final Technical Report](codes/DrishtiAI_Final_Report_and_Presentation_Script.md) but are explicitly marked there as **illustrative/expected**, pending completed LMMS-Eval runs — they are not reproduced here as confirmed results.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Android Studio (Giraffe or newer recommended) with NDK installed, for building the app
- An Android device or emulator running Android 10+ (Android 14 used for benchmarking)
- (Optional) An OpenAI API key, if using GPT-4o as the dataset-generation teacher model instead of local Qwen

### 1. Clone and set up the dataset pipeline

```bash
git clone <this-repo-url>
cd DL_Group-4_BLV/codes

pip install opencv-python tqdm openai
# For HuggingFace dataset output:
pip install datasets pyarrow
# For local Qwen VLM (no API cost):
pip install transformers accelerate qwen-vl-utils
# For CLIP-based frame selection:
pip install transformers torch torchvision
```

### 2. Run the full pipeline

**Cloud teacher (GPT-4o), cheapest query model:**

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

**Fully local, no API cost:**

```bash
python run_pipeline.py \
  --video_dir /data/videos \
  --output_root ./out \
  --vlm_backend qwen \
  --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
  --query_backend local \
  --n_pairs 20
```

**Estimate cost before running anything:**

```bash
python run_pipeline.py --video_dir /data/videos --output_root ./out --estimate_only
```

See [`codes/README.md`](codes/README.md) for the full step-by-step usage of each pipeline stage individually, output format reference, and resume/skip-steps support.

### 3. Fine-tune the model

```bash
pip install -r requirements_train.txt
python train_smolvla.py   # LoRA fine-tuning on the generated dataset
```

### 4. Evaluate

```bash
pip install -r requirements_eval.txt
python evaluate_blv.py    # Computes MCF / NAF on model outputs
```

### 5. Build and run the Android app

```bash
cd DrishtiAI
# Open the project in Android Studio, let Gradle sync,
# then build and deploy to a connected device or emulator.
```

The app requires the fine-tuned model weights (ONNX/GGUF format) to be placed in the appropriate assets directory before building — see in-app model management (`ModelsRepository.kt`) and the progress report's architecture section for exact paths.

---

## Configuration

Key Live Navigation tuning constants (in `LiveNavigationActivity.kt` and related files):

| Constant              |       Value | Purpose                                   |
| --------------------- | ----------: | ----------------------------------------- |
| Navigation window     |     1000 ms | Frame collection duration per cycle       |
| Sample interval       |      150 ms | Camera sampling rate during collection    |
| Max candidate frames  |           6 | Pool size before MobileNet selection      |
| Frames per grid       |           4 | Composited into a 2×2 image               |
| Tile size             |  192×192 px | Per-frame size after center-crop + resize |
| Composite grid size   |  384×384 px | Matches SigLIP's native input resolution  |
| Context size          | 2048 tokens | LLM context window                        |
| Max navigation output |   44 tokens | Keeps responses short and TTS-friendly    |
| Temperature           |        0.45 | Sampling temperature                      |
| Min-p                 |        0.05 | Nucleus-style sampling cutoff             |

---

## Performance Benchmarks

Measured on a **MediaTek Dimensity 7200** (4nm, Cortex-A715 + A510), 8GB LPDDR5 RAM, Android 14, ambient ~24°C, unplugged, 30-minute sustained session.

### Latency

| Stage                     |     Before |     After | Improvement |
| ------------------------- | ---------: | --------: | ----------: |
| Frame capture & composite |        ~2s |     ~0.8s |        2.5× |
| ONNX model inference      |    ~28–33s |    ~8–10s |       ~3.3× |
| TTS output                |      ~0.5s |     ~0.3s |       minor |
| **End-to-end cycle**      | **30–35s** | **9–11s** |     **~3×** |

**What drove the improvement:**

1. NNAPI delegate offloading INT4 ops to the device NPU (APU 650)
2. INT4 quantization of the LM head (8× memory bandwidth reduction vs. FP32)
3. Frame-drop during inference, eliminating queuing delay
4. Input resolution capped at 384×384 (down from 512×512, ~44% fewer vision-encoder FLOPs)
5. ONNX session warm-up at app launch, removing cold-start JIT cost

### Memory & thermal

| Metric                        | Value                           |
| ----------------------------- | ------------------------------- |
| Peak app RAM                  | ~2.0 GB                         |
| Model weights in memory       | ~310 MB (INT4 quantized)        |
| Frame buffer (4-frame grid)   | ~18 MB                          |
| CPU throttling observed       | None across a 30-minute session |
| Device heating                | Not noticeable                  |
| Battery draw during inference | ~8–10% per 10-minute session    |

### App size

| Component                |        Size |
| ------------------------ | ----------: |
| SmolVLM2-500M ONNX model |     ~410 MB |
| App code & resources     |      ~18 MB |
| Android runtime libs     |      ~17 MB |
| **Total**                | **~445 MB** |

> Future optimization path: ONNX model sharding + Play Asset Delivery to defer model download post-install, reducing initial install size to ~35MB.

Full benchmark methodology and additional ablations are in [`DrishtiAI_Progress_Report.md`](codes/DrishtiAI_Progress_Report.md).

---

## Known Limitations & Future Work

| Item                     | Status / Notes                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| App size (~445MB)        | Reducible via Play Asset Delivery for model files                                                    |
| Latency (9–11s)          | Further reducible via model distillation or speculative decoding (not yet implemented)               |
| Directional bias         | Significantly reduced but not fully eliminated at edge cases                                         |
| Connectivity             | Offline-only by design; no cloud fallback                                                            |
| Language support         | English TTS only; multilingual planned                                                               |
| Accessibility audit      | Full screen-reader / switch-access audit still pending                                               |
| Public benchmark results | LMMS-Eval (EgoSchema/PerceptionTest/Video-MME) numbers are illustrative pending completed runs       |
| User testing             | Safety-critical navigation needs real BLV user testing and failure-mode review before production use |

---

## Tech Stack

**Dataset & training (Python):**
PyTorch · Transformers · OpenCV · Qwen2.5-VL · OpenAI API · HuggingFace `datasets` · PyArrow

**Training hardware (reference spec from `requirements.txt`):**
8× NVIDIA RTX A6000 (49GB each), CUDA 12.1+, 20TB storage

**Android application:**
Kotlin · CameraX · ONNX Runtime Mobile · Android NNAPI · `llama.cpp` (C++ / JNI) · Material 3 · Android TTS

---

## Reports & Further Reading

This README summarizes the project; for full depth, see:

- [`codes/README.md`](codes/README.md) — dataset pipeline quick-start and CLI reference
- [`codes/DrishtiAI_Progress_Report.md`](codes/DrishtiAI_Progress_Report.md) — detailed engineering/performance report (Android optimization, benchmarks)
- [`codes/DrishtiAI_Final_Report_and_Presentation_Script.md`](codes/DrishtiAI_Final_Report_and_Presentation_Script.md) — full technical report, evaluation design, and presentation script
- [`codes/DrishtiAI_Technical_Report.tex`](codes/DrishtiAI_Technical_Report.tex) — LaTeX technical report source

---

## Team & Acknowledgements

Developed as **Group 4**'s deep learning project (BLV navigation track). Built on top of `SmolVLM2` (Hugging Face), `llama.cpp`, the Charades and AVCaps datasets, and Qwen2.5-VL as a dataset-generation teacher model.

---

## License

No license file is currently included in this repository. All rights reserved by the authors unless a license is added — if you intend to open-source this project, add a `LICENSE` file (e.g. MIT, Apache 2.0) at the repository root.
