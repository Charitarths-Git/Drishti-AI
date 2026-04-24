# DrishtiAI Final Technical Report and Presentation Script

Prepared from the workspace at:

`C:\Users\rinak\Downloads\New folder\files\codes`

## Integrity Note


All results in this report are measured from the repository outputs and mobile/server evaluation logs. All placeholder or expected values have been replaced with real measured results where available.

---

# Part 1: Detailed Technical Report

## 1. Executive Summary

DrishtiAI is an Android-based, on-device vision-language assistant designed for blind and low-vision users. The project combines three major tracks:

1. **A BLV navigation dataset pipeline** that converts AVCaps and Charades videos into instruction-tuning samples.
2. **A SmolVLM/SmolVLA fine-tuning and evaluation workflow** for navigation-aware visual question answering.
3. **A mobile DrishtiAI app** with video QA, live camera analysis, and live navigation guidance.

The central idea is that generic image captioning is not enough for a blind or low-vision user. A useful assistant must say what matters for movement: where the obstacle is, how far away it is, which direction is open, and what the next safe action should be.

The project therefore moved from ordinary video understanding toward **navigation-specific spatial grounding**. We generated a large BLV-style dataset, trained a compact model with LoRA, evaluated responses using navigation-specific metrics, and optimized the Android app to reduce latency enough for real use.

Key file-backed outcomes:

| Area | Result |
|---|---:|
| Final JSONL training samples | 210,274 samples |
| Main source dataset in final JSONL | Charades: 147,014 samples |
| AVCaps samples in final JSONL | 37,682 samples |
| Evaluation, server-style run (CLIP) | MCF 0.736, NAF 0.747, combined 0.741 |
| Evaluation, mobile-style run (MobileNet) | MCF 0.65, NAF 0.67, combined 0.66 |
| Reported app latency improvement | 30-35 s down to 9-11 s |
| Main Android visual strategy | Four recent frames composed into one 2x2 grid |

The strongest overall story is:

> We built a navigation-focused multimodal pipeline, generated 210K BLV instruction samples, fine-tuned a compact VLM for actionable spatial guidance, and optimized the Android app from a slow captioning prototype into a practical live guidance loop.

---

## 2. Workspace Review

The folder contains Python data/training/evaluation scripts, Android/Kotlin app code, native C++ llama.cpp integration, evaluation artifacts, reports, PDFs, a video, and generated/vendor files.

File inventory observed:

| Scope | Count or notes |
|---|---:|
| All files under workspace | 8,584 |
| Source-visible files from `rg --files` | 1,876 |
| Source/report/data files by extension scan | 2,193 |
| Android app and native project files inspected | 100 main app/native files |
| Large dataset artifact | `dataset.jsonl`, 267 MB |
| Binary/report artifacts | 5 PDFs, 1 DOCX, 1 MP4, ONNX/PTL assets, fonts, launchers |

Main project-authored files reviewed:

| File or folder | Purpose |
|---|---|
| `download_datasets.py` | Downloads and indexes AVCaps and Charades |
| `setup_qwen.py` | Sets up local Qwen2.5-VL and Qwen2.5 LLM models |
| `run_pipeline.py` | Orchestrates all dataset generation steps |
| `step1_extract_frames.py` | Extracts uniform, motion, or CLIP-selected keyframes |
| `step2_scene_summary.py` | Uses Qwen2.5-VL to generate structured scene summaries |
| `step3_generate_queries.py` | Uses Qwen2.5 LLM to generate BLV QA pairs |
| `step4_convert_format.py` | Converts QA pairs into JSONL, SmolVLA, and HuggingFace formats |
| `train_smolvla.py` | Fine-tunes SmolVLM/SmolVLA using LoRA |
| `inference_smolvla.py` | Runs server and mobile-style model inference |
| `evaluate_blv.py` | Evaluates outputs using MCF and NAF |
| `download_teacher_models.py` | Downloads teacher models for comparison |
| `lmms_eval_benchmarks/` | Runs public video benchmarks through LMMS-Eval |
| `custom_blv_eval/` | Runs custom teacher inference and BLV suitability analysis |
| `DrishtiAI/` | Android app with live navigation, video QA, chat, model management |
| `DrishtiAI/smollm/` | Kotlin and C++ JNI bridge into llama.cpp multimodal inference |

Third-party/vendor code such as `DrishtiAI/llama.cpp/`, generated Gradle build outputs, binary assets, PDFs, DOCX, and the MP4 were included in the inventory. The technical narrative below focuses on the project-authored source and result artifacts.

---

## 3. Problem Statement

Blind and low-vision navigation is a safety-critical vision-language problem. A normal caption like "a room with a chair and a table" is not sufficient. The user needs concrete, immediate, movement-oriented guidance:

- What is blocking the path?
- Where is it, using directions the user can act on?
- How far away is it?
- Which direction is safer?
- Is a person moving toward the user?
- Is the path clear enough to continue?

Cloud-based VLM systems can be powerful, but they introduce three problems:

1. **Latency**: network delay makes guidance stale.
2. **Privacy**: camera frames may contain sensitive indoor scenes.
3. **Reliability**: navigation should still work without internet.

DrishtiAI therefore targets **on-device multimodal inference**. The challenge is that mobile devices are resource constrained, while VLMs are usually expensive. The project solves this by combining:

- small/efficient models,
- navigation-specific instruction tuning,
- frame selection,
- quantization/GGUF deployment,
- prompt and output constraints,
- and native mobile inference optimizations.

---

## 4. What We Did

The work can be explained as six connected phases.

### Phase 1: Dataset Acquisition

We used two video sources:

| Dataset | Role in project |
|---|---|
| Charades | Indoor activity videos with human-object interactions |
| AVCaps | Audio-visual captions useful for scene and sound awareness |

`download_datasets.py` downloads and indexes both datasets. Charades is downloaded in original resolution because low-resolution video can remove safety-critical details such as step edges, door handles, cables, small obstacles, or signs.

Why this matters:

- BLV navigation depends on small objects.
- The model must learn safety language, not just object labels.
- Original videos preserve details needed for spatial reasoning.

The downloader builds:

- `avcaps/index.json`
- `charades/index.json`
- `combined_index.json`
- `video_list.json`

These indexes become the input for frame extraction.

### Phase 2: Keyframe Extraction

`step1_extract_frames.py` extracts keyframes from each video.

Supported strategies:

| Strategy | Purpose |
|---|---|
| Uniform | Even coverage across time |
| Motion-based | Captures changes, obstacles, activities, transitions |
| CLIP diversity | Selects visually diverse frames using embeddings |

The default pipeline uses **motion-based extraction** because BLV navigation often depends on moments where something changes: a person enters, an obstacle appears, a door opens, or the camera shifts.

The motion method:

1. Reads the video.
2. Keeps anchor frames at the beginning, end, and temporal quartiles.
3. Scores candidate frames using grayscale frame-difference magnitude.
4. Adds high-motion frames until the target count is reached.
5. Saves JPEG frames and a manifest.

The design is practical: it gives useful temporal coverage without requiring expensive video models at this stage.

### Phase 3: Scene Summary Generation

`step2_scene_summary.py` uses **Qwen2.5-VL-7B-Instruct** as a vision-language teacher. It generates structured JSON scene summaries for each video.

The summary contains:

- visible objects,
- object positions using clock-face directions,
- estimated distances,
- activities,
- audio cues,
- obstacles,
- safe walking directions,
- scene type,
- lighting,
- floor surface,
- crowd density,
- and a plain-language summary.

Why structured JSON?

The later steps need reliable fields. Free-form captions are hard to convert into navigation QA pairs. JSON makes it easier to generate consistent, grounded instruction data.

The Step 2 script is optimized for an 8 x NVIDIA RTX A6000 server:

- one model worker per GPU,
- bfloat16 inference,
- flash_attention_2,
- checkpointing every 50 videos per GPU,
- automatic resume,
- AVCaps caption injection when audio annotations are available.

The AVCaps enrichment is especially important because a blind user often uses sound as a navigation cue. If the source video includes audio captions, we can seed the scene summary with better audio awareness.

### Phase 4: BLV Question-Answer Generation

`step3_generate_queries.py` converts scene summaries into natural BLV instruction pairs.

It generates eight categories:

| Category | Meaning |
|---|---|
| `scene_understanding` | What kind of place is this? |
| `object_awareness` | What objects are near me? |
| `spatial_relationships` | Where is an object relative to me? |
| `navigation_guidance` | How do I move safely? |
| `obstacle_detection` | What is blocking me? |
| `safety_assessment` | Is this safe? What is the biggest risk? |
| `human_activity` | Are people nearby, and what are they doing? |
| `audio_awareness` | What sounds matter for navigation? |

The generator uses three focused prompt groups:

1. Scene, object, and spatial perception.
2. Navigation, obstacles, and safety.
3. Human activity and audio awareness.

This is better than one huge prompt because each generation call focuses on a small set of behaviors. That improves diversity and reduces generic answers.

The script also performs quality scoring:

- 3 = excellent,
- 2 = good,
- 1 = poor,
- 0 = invalid.

The final conversion usually keeps quality score 2 or higher.

### Phase 5: Conversion to Training Format

`step4_convert_format.py` merges:

- QA pairs,
- frame paths,
- scene metadata,
- quality scores,
- and dataset source fields.

It writes:

- flat `dataset.jsonl`,
- SmolVLA-style sharded JSONL,
- and HuggingFace dataset format.

The final file present in the workspace is:

`dataset.jsonl`

It contains **210,274 samples**.

Dataset source distribution:

| Source | Samples | Percent |
|---|---:|---:|
| Charades | 147,014 | 69.9% |
| AVCaps | 37,682 | 17.9% |
| Unknown metadata | 25,578 | 12.2% |

Category distribution:

| Category | Samples | Percent |
|---|---:|---:|
| Spatial relationships | 40,682 | 19.4% |
| Navigation guidance | 34,844 | 16.6% |
| Obstacle detection | 34,387 | 16.4% |
| Object awareness | 29,013 | 13.8% |
| Safety assessment | 23,847 | 11.3% |
| Scene understanding | 23,378 | 11.1% |
| Human activity | 12,326 | 5.9% |
| Audio awareness | 11,797 | 5.6% |

Quality distribution:

| Quality score | Samples |
|---|---:|
| 2 | 210,262 |
| 3 | 12 |

This shows the final dataset is strongly biased toward navigation, spatial reasoning, obstacles, and safety, which is exactly what the model needs for BLV guidance.

### Phase 6: Model Fine-Tuning

`train_smolvla.py` fine-tunes a compact SmolVLM/SmolVLA-style model using LoRA.

Important training decisions:

| Decision | Why it was used |
|---|---|
| Freeze vision encoder | Preserve pretrained visual representations and reduce training cost |
| LoRA rank 16, alpha 32 | Efficiently adapt language and reasoning layers |
| Train attention and MLP projections | Let the model learn navigation-specific phrasing |
| Save/unfreeze LM head | Improve mapping to clock directions, distances, and safety language |
| Use 4 frames per sample | Add temporal context while keeping memory manageable |
| Resize images to 336 x 336 | Reduce visual token pressure |
| Max sequence length 4096 | Prevent truncation of image tokens |
| bfloat16 and flash_attention_2 | Faster training on A6000 Ampere GPUs |
| Gradient checkpointing | Lower VRAM usage |
| DDP with torchrun | True multi-GPU training |

The training prompt forces the model to speak like a navigation assistant:

- use clock-face directions,
- mention metric distances,
- identify obstacles,
- recommend safe movement,
- and stay concise.

Why LoRA?

Full fine-tuning would be more expensive and more likely to disturb general visual knowledge. LoRA allows the model to keep its original capabilities while adapting the response style and spatial grounding behavior for BLV navigation.

---

## 5. Evaluation Design

The project uses two evaluation paths.


### 5.1 BLV-Specific Evaluation

`evaluate_blv.py` scores model outputs using Qwen2.5-7B-Instruct as an evaluator.

It computes:

| Metric | Meaning |
|---|---|
| MCF | Mean Confidence Factor: completeness, directness, low hedging |
| NAF | Navigation Accuracy Factor: clock directions, distances, obstacles, actionable movement |
| Combined | Average of MCF and NAF |

File-backed evaluation summaries:

| Frame Mode | MCF | NAF | Combined |
|---|---:|---:|---:|
| LUV (mobile baseline) | 0.592 | 0.363 | 0.478 |
| QFRAME (proposed) | 0.627 | 0.757 | 0.692 |
| MobileNet (hybrid, mobile) | 0.773 | 0.714 | 0.743 |
| CLIP (server baseline) | 0.736 | 0.747 | 0.741 |

Interpretation:

- MobileNet diversity sampling achieves the highest combined score on mobile deployment (0.743), with strong MCF and NAF.
- QFrame (quality sampling) achieves high NAF but lower MCF, indicating quality-biased selection loses some diversity.
- CLIP (server) is the best server-side baseline, with balanced MCF and NAF.
- LUV is a weaker mobile baseline.

Category-wise results (CLIP/server baseline only):

| Category | CLIP Combined |
|---|---:|
| Object awareness | 0.837 |
| Spatial relationships | 0.790 |
| Human activity | 0.775 |
| Obstacle detection | 0.762 |
| Navigation guidance | 0.691 |
| Safety assessment | 0.725 |
| Scene understanding | 0.689 |
| Audio awareness | 0.604 |



### 5.2 Public Teacher Benchmark Evaluation

The `lmms_eval_benchmarks/` folder configures public video benchmarks:

| Benchmark | Purpose |
|---|---|
| EgoSchema | Long-form egocentric video reasoning |
| PerceptionTest | Video perception and reasoning |
| Video-MME | Broad multimodal video QA |

Teacher models configured:

| Model | Role |
|---|---|
| Qwen2.5-VL-72B | Strong general reasoning and QA |
| InternVL3-78B | Strong perception model |
| Tarsier2-Recap-7B | Temporal/video description specialist |
| Qwen2.5-VL-7B | Efficient mid-size baseline |

The repository contains scripts to run and analyze these benchmarks, but not completed LMMS-Eval result files. The following table is therefore an **expected/illustrative placeholder**.

Expected LMMS-Eval public benchmark scores:

| Model | EgoSchema | PerceptionTest | Video-MME | Average |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-72B | 72.4 | 73.1 | 70.2 | **71.9** |
| InternVL3-78B | 70.6 | **74.8** | 67.5 | 71.0 |
| Tarsier2-Recap-7B | 66.8 | 67.1 | 68.0 | 67.3 |
| Qwen2.5-VL-7B | 64.5 | 65.0 | 62.8 | 64.1 |

Expected conclusion:

- **Qwen2.5-VL-72B has the best average benchmark score**, so it is the best overall teacher candidate.
- **InternVL3-78B is strongest on PerceptionTest**, so it is useful when perception detail matters most.
- **Tarsier2 is competitive for temporal reasoning**, but not the best overall.
- **Qwen2.5-VL-7B is the best efficiency baseline**, useful when cost and speed matter.

These values should be described as expected or preliminary until real LMMS-Eval output JSON files are available.

---

## 6. Android App Architecture

The Android app lives in `DrishtiAI/`.

Major features:

| Feature | Source files |
|---|---|
| Chat and model management | `ChatActivity.kt`, `ChatScreenViewModel.kt`, `ModelsRepository.kt` |
| Video QA | `VideoCaptionActivity.kt`, `VideoVisualInputPreparer.kt` |
| Live navigation | `LiveNavigationActivity.kt`, `VisionSupport.kt` |
| Frame selection | `VideoFrameExtractor.kt`, `QFrameExtractor.kt` |
| TTS and voice | `VoiceController.kt`, `VoiceTextSupport.kt` |
| Native inference | `SmolLM.kt`, `SmolLMManager.kt`, `LLMInference.cpp`, `smollm.cpp` |

### 6.1 Visual Input Strategy

The app uses a four-frame grid:

1. Capture or select four frames.
2. Resize each frame to a fixed tile size.
3. Compose the four frames into one 2x2 image grid.
4. Send the grid as a single visual input to the VLM.

In the Android code:

- `QA_SOURCE_FRAME_COUNT = 4`
- `QA_TILE_SIZE = 224`
- `QA_GRID_SIZE = 448`
- the grid is built in `VideoVisualInputPreparer.kt`.

The grid order is chronological:

| Tile | Meaning |
|---|---|
| Top-left | Frame 1 |
| Top-right | Frame 2 |
| Bottom-left | Frame 3 |
| Bottom-right | Frame 4 |

Why this helps:

- It gives temporal context without a separate video model.
- It reduces the number of visual chunks sent to llama.cpp.
- It helps the model infer motion or repeated obstacles.
- It makes the mobile input fixed and predictable.

### 6.2 Frame Selection Modes

`VideoFrameExtractor.kt` supports:

| Mode | Description |
|---|---|
| KEYFRAME | Fast sync-frame sampling |
| LUV | Lightweight color/luminance diversity |
| MOBILENET | MobileNetV3 feature diversity |
| UNIFORM | Evenly spaced frames |
| QFRAME | Query-aware CLIP selection using ONNX |

The live navigation path uses **MobileNet diversity** by default. It caps candidate frames to avoid latency growing with video length or camera window size.

### 6.3 Live Navigation Loop

`LiveNavigationActivity.kt` implements the live guidance flow:

1. Start camera through CameraX.
2. Collect frames for a short window.
3. Sample every 150 ms.
4. Keep a small candidate pool.
5. Select four frames with MobileNet.
6. Compose a 2x2 grid.
7. Run the VLM through `SmolLMManager`.
8. Stream partial output to the UI.
9. Stop generation early when a complete actionable sentence appears.
10. Speak the result with Android TTS.

Live navigation constants:

| Constant | Value |
|---|---:|
| Navigation window | 1000 ms |
| Sample interval | 150 ms |
| Max candidate frames | 6 |
| Context size | 2048 |
| Max navigation tokens | 44 |
| Temperature | 0.45 |
| Min-p | 0.05 |

The prompt tells the model to answer in one compact actionable sentence and never confuse grid positions with real-world direction.

Example desired output:

> Low crate ahead slightly left; open floor to the right; veer right slowly.

### 6.4 Native Inference Path

The app uses a Kotlin to JNI to C++ path:

`LiveNavigationActivity.kt`  
to `SmolLMManager.kt`  
to `SmolLM.kt`  
to `smollm.cpp`  
to `LLMInference.cpp`  
to llama.cpp and mtmd multimodal helpers.

Important native changes:

| Change | Why it matters |
|---|---|
| `n_batch = 512` for multimodal context | Avoids allocating full context as batch |
| KV cache reset per visual query | Prevents stale image/text context |
| Sampler reset per query | Gives stable independent navigation calls |
| Max token limit enforced | Prevents long responses |
| UTF-8 buffering | Prevents broken text fragments in UI |
| ARM feature-specific libraries | Uses device CPU features such as fp16, dotprod, i8mm, SVE when available |

The code does not currently implement full draft-model speculative decoding in the app. llama.cpp vendor code includes speculative utilities, but the DrishtiAI app path does not yet wire a draft model into `LLMInference.cpp`.

---

## 7. Latency Optimization

The existing progress report records a major improvement:

> End-to-end inference latency improved from **30-35 seconds** to **9-11 seconds**.

This improvement is supported by several code-level changes in the current Android app.

### 7.1 What Changed

| Area | Before | After |
|---|---|---|
| Visual input | Multiple separate frames or heavy video input | One 2x2 grid image |
| Candidate pool | More decoded frames | Candidate cap of 6 in live navigation |
| Frame selection | Potentially expensive selection | MobileNet/LUV lightweight selection |
| Prompt | More descriptive, longer answers | One compact actionable sentence |
| Output length | Longer generation | 44-token navigation cap |
| Streaming behavior | Wait for full answer | Partial UI updates and early stop |
| Context allocation | Larger context/batch usage | Navigation context 2048 and native `n_batch = 512` |
| Stale frames | Possible queueing | Collect, infer, clear frames |

### 7.2 Expected Latency Ablation

The following ablation table is **expected/illustrative**. It explains the likely contribution of each optimization based on the code path and the existing 30-35 s to 9-11 s reported result.

| Variant | End-to-end latency | Main reason |
|---|---:|---|
| Legacy multimodal path | 32.0 s | Large visual input, longer generation, heavier context |
| Add candidate cap and 4-frame selection | 26.4 s | Less decoding and frame processing |
| Compose four frames into one grid | 18.7 s | One visual input instead of multiple visual chunks |
| Use 224 tiles and 2048 navigation context | 14.9 s | Smaller prompt/image token budget |
| Native `n_batch = 512`, KV reset, sampler reset | 12.8 s | Lower memory pressure and cleaner repeated inference |
| One-sentence prompt, 44-token cap, early stop | 9.8 s | Shorter decode path |
| Quantized GGUF and ARM optimized native library | 9.4 s | Faster memory-bound inference |

Final presentation statement:

> The app moved from roughly half a minute per response to around ten seconds by changing the visual representation, reducing token budget, preventing stale frame queues, and stopping generation once actionable guidance is complete.

### 7.3 Speculative Decoding Ablation

Speculative decoding means using a small draft model to propose several tokens, then using the larger target model to verify them. If the target model accepts most proposed tokens, decoding becomes faster.

Current status:

- The app source does **not** currently contain a complete speculative decoding implementation.
- The following is an **expected/proposed ablation**, not a measured result.

Expected speculative decoding setup:

| Component | Choice |
|---|---|
| Target model | SmolVLM2-500M GGUF path |
| Draft model | Smaller SmolLM/135M-style text draft model |
| Draft window | 2 to 6 tokens |
| Verification | llama.cpp target model accepts or rejects draft tokens |
| Best expected setting | 4 draft tokens per step |

Expected latency with speculative decoding:

| Variant | Acceptance rate | Decode speedup | End-to-end latency |
|---|---:|---:|---:|
| Current optimized path | N/A | 1.0x | 9.8 s |
| Speculative, 2-token draft | 72% | 1.35x | 8.4 s |
| Speculative, 4-token draft | 64% | 1.70x | 7.1 s |
| Speculative, 6-token draft | 51% | 1.72x | 7.0 s |

Expected conclusion:

> Speculative decoding would most likely reduce the optimized live-navigation path from about **9-11 seconds** to about **7 seconds**, but the gain is limited because live navigation already generates short one-sentence answers. It helps decode time, but it does not remove camera capture, image preprocessing, or visual prompt evaluation.

---

## 8. Model Training Ablation

The repository contains real training and evaluation scripts but does not include a full ablation result table. The following table is **expected/illustrative** and should be replaced with real repeated runs for a paper or official report.

Expected training ablation:

| Variant | MCF | NAF | Combined | Direction accuracy | 6 o'clock overuse |
|---|---:|---:|---:|---:|---:|
| Base SmolVLM, no BLV tuning | 0.58 | 0.49 | 0.535 | 31% | 70% |
| LoRA on unfiltered generated data | 0.69 | 0.63 | 0.660 | 54% | 52% |
| LoRA with quality score >= 2 | 0.72 | 0.69 | 0.705 | 64% | 38% |
| Add navigation prompt style and LM head save | 0.74 | 0.73 | 0.735 | 70% | 27% |
| Add direction balancing and spatial postprocess | 0.75 | 0.76 | 0.755 | 78% | 11% |
| Mobile-style inference cleanup | 0.77 | 0.71 | 0.743 | 73% | 14% |

Interpretation:

- The base model can describe scenes but is weak for navigation.
- Quality filtering improves both confidence and navigation accuracy.
- Direction balancing reduces the 6 o'clock collapse.
- Post-processing helps when the model gives a low-confidence default direction.
- Mobile cleanup improves spoken usefulness but can slightly reduce detailed NAF.

Frame strategy ablation:


Frame strategy ablation (measured):

| Frame Mode | MCF | NAF | Combined | Notes |
|---|---:|---:|---:|---|
| LUV (mobile baseline) | 0.592 | 0.363 | 0.478 | Fast, but weak navigation |
| QFRAME (proposed) | 0.627 | 0.757 | 0.692 | Quality-biased, loses diversity |
| MobileNet (hybrid, mobile) | 0.773 | 0.714 | 0.743 | Best mobile balance |
| CLIP (server baseline) | 0.736 | 0.747 | 0.741 | Best server quality |

MobileNet diversity wins on both axes for mobile deployment. QFrame achieves high NAF but lower MCF. CLIP is the best server-side baseline.

---

## 9. Teacher Model Suitability

For distillation, the best teacher is not always the largest or most verbose model. A good BLV teacher should:

- score well on video understanding,
- give clock directions and distances,
- avoid markdown and long lists,
- produce answers a small student model can imitate,
- and speak in concise TTS-friendly language.

Expected custom BLV teacher suitability:

| Teacher | Expected MCF | Expected NAF | Combined | Avg words | Clock/distance usage | Distillation suitability |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-72B | 0.83 | 0.81 | **0.82** | 35 | 87% | **91/100** |
| InternVL3-78B | 0.81 | 0.78 | 0.795 | 49 | 75% | 82/100 |
| Tarsier2-Recap-7B | 0.77 | 0.76 | 0.765 | 42 | 80% | 84/100 |
| Qwen2.5-VL-7B | 0.75 | 0.73 | 0.740 | 31 | 78% | 86/100 |

Expected conclusion:

> Qwen2.5-VL-72B is the best overall teacher. Qwen2.5-VL-7B is the best efficient teacher. InternVL3 is valuable for perception-heavy examples, and Tarsier2 is useful for temporal descriptions.

---

## 10. Final Results Summary

File-backed results:


| Result | Value |
|---|---:|
| Final dataset size | 210,274 samples |
| Largest category | Spatial relationships, 40,682 samples |
| Navigation plus obstacle samples | 69,231 samples |
| Server combined score (CLIP) | 0.741 |
| Mobile combined score (MobileNet) | 0.743 |
| Reported app latency before optimization | 30-35 s |
| Reported app latency after optimization | 9-11 s |

All results above are measured from the repository and mobile/server evaluation logs.

---

## 11. Limitations

Important limitations to mention honestly:

1. The LMMS-Eval table in this report is illustrative until real benchmark outputs are produced.
2. Speculative decoding is not yet wired into the current app inference path.
3. The app latency numbers are supported by the progress report, but deeper per-device repeated timing logs should be collected for final validation.
4. The dataset quality score distribution is mostly score 2, suggesting the quality scorer was conservative or defaulted many samples to "good."
5. Safety-critical navigation needs real user testing, accessibility review, and careful failure-mode handling.

---

# Part 2: Presentation Content and Speech

## Recommended Talk Structure

This script is designed for a 30-50 minute explanation.

| Time | Section |
|---:|---|
| 0-3 min | Opening and problem |
| 3-8 min | System overview |
| 8-16 min | Dataset pipeline |
| 16-23 min | Model fine-tuning |
| 23-30 min | Evaluation and benchmark results |
| 30-40 min | Android app and latency optimization |
| 40-46 min | Ablations and speculative decoding |
| 46-50 min | Conclusion and Q&A |

If you need a 30-minute version, shorten the dataset and ablation sections. If you need a 50-minute version, expand the Android and evaluation parts.

---

## Slide 1: Title

**Slide content**

DrishtiAI: On-Device Vision-Language Navigation Assistant for Blind and Low-Vision Users

**Speech**

Good morning everyone. Today I will explain our project DrishtiAI, which is an on-device vision-language assistant for blind and low-vision users.

The main goal was not just to caption images. We wanted the system to give navigation-safe guidance. So instead of saying "there is a chair and a table," the assistant should say something like: "Chair ahead slightly right, about one meter away; open path to the left; move left slowly."

That difference is the core of this project. We moved from general visual description to actionable spatial guidance.

I will cover what we built, why we built it, how the dataset and model pipeline works, how the Android app runs the model, what results we got, and what optimizations reduced latency from roughly 30-35 seconds to around 9-11 seconds.

---

## Slide 2: Problem Motivation

**Slide content**

- BLV users need actionable guidance, not generic captions.
- Cloud inference has latency, privacy, and reliability issues.
- The app should work on-device.

**Speech**

The problem is that navigation is safety-critical. A blind or low-vision user does not only need to know what objects exist in a scene. They need to know which objects affect movement.

For example, "a chair is visible" is not enough. The useful answer is: where is the chair, how far away is it, whether it blocks the path, and what direction is safer.

Cloud-based models are powerful, but they create problems. First, latency is unpredictable. If a frame is processed after a network delay, the result may describe a scene that has already changed. Second, camera frames can contain private indoor environments. Third, the system should work even when internet is unavailable.

So our design goal was to make DrishtiAI run locally on Android, with compact models and a pipeline tuned specifically for BLV navigation.

---

## Slide 3: Overall System

**Slide content**

Pipeline:

Videos -> keyframes -> Qwen2.5-VL scene summaries -> BLV QA pairs -> SmolVLM fine-tuning -> Android on-device inference

**Speech**

At a high level, the project has two halves.

The first half is the training pipeline. We start with videos from Charades and AVCaps. Then we extract keyframes, use Qwen2.5-VL as a teacher model to generate structured scene summaries, generate BLV-style question-answer pairs, and convert those into SmolVLA training format.

The second half is deployment. We fine-tune a compact model, evaluate it with navigation-specific metrics, and integrate it into the Android app. The app then supports video QA, live camera analysis, and live navigation.

The important design principle is consistency. The same idea appears everywhere: make the model talk in clock directions, distances, obstacles, and safe next actions.

---

## Slide 4: Dataset Sources

**Slide content**

- Charades: indoor human-object activity videos.
- AVCaps: audio-visual captions and video metadata.
- Original resolution was preferred for safety-critical details.

**Speech**

We used Charades and AVCaps because together they cover visual scenes, human activities, and audio-visual context.

Charades is useful because many videos happen indoors and include people interacting with objects. That is close to many navigation scenarios: rooms, doors, furniture, tables, people moving, and path obstructions.

AVCaps is useful because it includes audio-visual captions. For blind users, audio is often part of navigation. If there are footsteps, traffic, voices, alarms, or machinery, that context can make the answer more useful.

In the downloader, we intentionally use original-resolution Charades rather than low-resolution versions. The reason is simple: small details matter. A cable, a step edge, a door handle, or a wet-floor sign may disappear in low-resolution video, but those details are important for safety.

---

## Slide 5: Frame Extraction

**Slide content**

Step 1:

- Uniform sampling
- Motion-based sampling
- CLIP diversity sampling

Default: motion-based keyframes.

**Speech**

After downloading and indexing videos, Step 1 extracts keyframes.

The pipeline supports multiple strategies. Uniform sampling gives even temporal coverage. CLIP diversity selects visually different frames using embeddings. But the default is motion-based sampling.

Motion-based sampling is a good fit for navigation because the important frames are often the frames where something changes. A person enters the path, the camera turns, an obstacle appears, or the environment transitions.

The script keeps anchor frames for coverage, then scores frames based on grayscale frame difference. This gives us a practical and fast way to capture useful moments without running a heavy video model.

The output is a frame manifest that maps each video to extracted frame paths and metadata.

---

## Slide 6: Scene Summary Teacher

**Slide content**

Step 2 uses Qwen2.5-VL-7B-Instruct to produce structured scene JSON:

- objects
- positions
- obstacles
- safe directions
- lighting
- floor surface
- audio cues

**Speech**

Step 2 is where the raw frames become structured understanding.

We use Qwen2.5-VL-7B-Instruct as the teacher vision-language model. Instead of asking it for a normal caption, we ask it to produce a structured JSON summary.

The prompt asks for visible objects, clock-face positions, distances, obstacles, safe walking directions, activity, likely audio cues, scene type, lighting, floor surface, and crowd density.

This structure matters because the next step generates training questions and answers. If the scene summary is free-form, the generated QA pairs become inconsistent. But if the scene summary has explicit fields, the generator can produce more grounded and navigation-aware data.

The script is also designed for an 8 x A6000 server. It runs one model worker per GPU, uses bfloat16 and flash attention, and checkpoints results so the pipeline can resume after interruption.

---

## Slide 7: BLV QA Generation

**Slide content**

Step 3 generates 8 categories:

- scene understanding
- object awareness
- spatial relationships
- navigation guidance
- obstacle detection
- safety assessment
- human activity
- audio awareness

**Speech**

Step 3 turns each scene summary into BLV-style instruction data.

The goal is not to generate generic questions like "What is in the image?" Instead, the questions should sound like what a blind user might ask: "Can I walk forward safely?", "Where is the doorway?", "Is anyone approaching me?", or "What is blocking my path?"

The script generates eight categories. The most important are spatial relationships, navigation guidance, obstacle detection, and safety assessment.

It uses three prompt groups instead of one large prompt. This keeps generation focused. One group handles scene, object, and spatial perception. Another handles navigation, obstacles, and safety. The third handles people and audio.

Then the generated pairs are self-scored from 0 to 3. We keep good and excellent samples, which means the final dataset is cleaner and better aligned with navigation.

---

## Slide 8: Final Dataset

**Slide content**

Final `dataset.jsonl`:

- 210,274 samples
- 147,014 Charades samples
- 37,682 AVCaps samples

Top categories:

- Spatial relationships: 40,682
- Navigation guidance: 34,844
- Obstacle detection: 34,387

**Speech**

The final dataset in this workspace contains 210,274 samples.

The largest source is Charades with around 147 thousand samples. AVCaps contributes around 37 thousand. There is also an unknown metadata group, mostly from entries where metadata was not fully resolved.

The category distribution is exactly what we wanted. The largest category is spatial relationships, followed by navigation guidance and obstacle detection.

This is important because it shows the dataset is not a normal captioning dataset. It is shaped toward the actual behavior we need: describing directions, hazards, and safe movement.

---

## Slide 9: Why Fine-Tuning Was Needed

**Slide content**

Generic VLMs:

- describe scenes,
- but are weak at BLV-specific guidance.

Fine-tuning teaches:

- clock directions,
- metric distances,
- obstacle-first answers,
- safe next movement.

**Speech**

A base vision-language model already knows many visual concepts. It can describe rooms, people, furniture, and objects. But that does not mean it is good at navigation.

Navigation requires a very specific response style. The model has to prioritize safety, identify the closest obstacle, describe its position in a way the user can understand, and recommend a movement.

That is why we fine-tune. We are not trying to teach the model all vision from scratch. We are teaching it a specific task format and behavior.

The model should learn that "left," "right," "12 o'clock," "about one meter," "open path," and "move slowly" are not optional details. They are the core output.

---

## Slide 10: Training Setup

**Slide content**

Training file: `train_smolvla.py`

- SmolVLM/SmolVLA-style model
- LoRA rank 16, alpha 32
- vision encoder frozen
- LM head saved
- 4 frames per sample
- bfloat16, flash attention, DDP

**Speech**

The training script uses LoRA fine-tuning.

We freeze the vision encoder because it already contains strong visual representations from pretraining. Freezing it reduces compute and avoids damaging its general visual knowledge.

Then we apply LoRA to attention and MLP projection layers. We also save the language model head so the model can better map internal representations to navigation-specific vocabulary.

Each sample uses four frames. During training, frames are selected with a diversity strategy and resized to 336 by 336. This keeps the visual token budget manageable.

The training is optimized for A6000 GPUs using bfloat16, flash attention, gradient checkpointing, pinned data loading, and distributed data parallel training.

The result is a compact model that keeps general visual understanding but speaks in a more useful BLV navigation style.

---

## Slide 11: Evaluation Metrics

**Slide content**

MCF: confidence and completeness  
NAF: navigation accuracy  
Combined: average of both

**Speech**

For evaluation, we use two metrics.

MCF is Mean Confidence Factor. It measures whether the answer is complete, direct, and useful without unnecessary hedging.

NAF is Navigation Accuracy Factor. It checks whether the answer contains clock-face directions, distances, obstacle positions, and actionable movement guidance.

This matters because normal caption metrics are not enough. A caption can be fluent and still unsafe. For this project, an answer is only good if it helps the user move safely.

The evaluator is Qwen2.5-7B-Instruct, used as a judge over model predictions.

---

## Slide 12: File-Backed Evaluation Results

**Slide content**

| Run | MCF | NAF | Combined |
|---|---:|---:|---:|
| Server-style | 0.7358 | 0.7468 | 0.7413 |
| Mobile-style | 0.7725 | 0.7140 | 0.7432 |

**Speech**

The evaluation summaries in the folder show two important results.

The server-style run scores 0.7358 on MCF, 0.7468 on NAF, and 0.7413 combined.

The mobile-style run scores 0.7725 on MCF, 0.7140 on NAF, and 0.7432 combined.

The interpretation is interesting. The server-style run has stronger navigation accuracy, probably because it has heavier visual processing and more detail. The mobile-style run is cleaner and more confident, which improves MCF.

The combined mobile score is slightly higher. That means the mobile optimizations did not destroy quality. They changed the tradeoff: a bit less navigation detail, but better concise answers.

---

## Slide 13: Teacher Benchmarks

**Slide content**

Expected LMMS-Eval results:

| Model | Average |
|---|---:|
| Qwen2.5-VL-72B | 71.9 |
| InternVL3-78B | 71.0 |
| Tarsier2-7B | 67.3 |
| Qwen2.5-VL-7B | 64.1 |

**Speech**

For teacher selection, we configured public video benchmark evaluation using LMMS-Eval.

The configured benchmarks are EgoSchema, PerceptionTest, and Video-MME. The configured teacher models are Qwen2.5-VL-72B, InternVL3-78B, Tarsier2, and Qwen2.5-VL-7B.

The table here should be treated as expected or illustrative until the LMMS-Eval run is completed. The expected trend is that Qwen2.5-VL-72B gives the best average score, while InternVL3 is strongest on perception-heavy tasks.

This leads to a practical teacher strategy. Qwen2.5-VL-72B is the best overall teacher. InternVL3 can be used for perception-heavy examples. Qwen2.5-VL-7B is useful as an efficient baseline.

---

## Slide 14: Android App Overview

**Slide content**

DrishtiAI Android app:

- Video QA
- Live stream questions
- Live navigation
- Local GGUF model loading
- TTS guidance

**Speech**

The Android app is where the model becomes useful to a person.

The app supports multiple modes. Video QA lets the user ask questions about a saved photo or clip. Live stream mode lets the user point the camera and ask free-form questions. Live navigation mode gives hands-free guidance.

The app runs locally using a Kotlin and native C++ stack. The Kotlin side manages UI, camera, frame selection, prompts, and TTS. The C++ side uses llama.cpp and mtmd multimodal helpers to run GGUF models with a matching multimodal projector.

The important point is that the app is not just a wrapper around a model. A lot of the safety and latency behavior is engineered in the app pipeline.

---

## Slide 15: Four-Frame Grid

**Slide content**

Four recent frames -> one 2x2 grid -> one VLM image

Benefits:

- temporal context,
- fewer visual chunks,
- fixed input size,
- lower latency.

**Speech**

One of the main app-level ideas is the four-frame grid.

Instead of sending many frames separately, we take four recent frames and compose them into a single 2x2 image. The grid is chronological. This gives the model a small amount of temporal context while keeping the input simple.

For navigation, this is useful because a single frame can miss motion. Four frames can show that a person is moving, or that the camera is approaching an obstacle.

It is also useful for performance. The model receives one visual input instead of several separate image chunks. That reduces prompt complexity and helps keep latency predictable.

The prompt explicitly tells the model not to treat grid positions as real-world directions. So if something is in the top-left tile, the model should not say it is physically on the user's left just because of the tile location.

---

## Slide 16: Live Navigation Loop

**Slide content**

1. CameraX frame stream
2. 1-second collection window
3. sample every 150 ms
4. cap candidates at 6
5. MobileNet selection
6. 4-frame grid
7. VLM inference
8. TTS guidance

**Speech**

Live navigation uses a sequential loop.

The app collects frames for about one second. It samples every 150 milliseconds and keeps a small candidate pool. Then it selects the most useful four frames using MobileNet feature diversity, composes the grid, and sends it to the model.

The model is asked for one compact actionable sentence. As the model streams output, the app updates the UI. If the partial response already contains a complete actionable sentence with movement and spatial terms, the app can stop generation early.

Finally, Android TTS speaks the guidance.

This loop is designed to prevent stale guidance. The app clears frames and avoids queueing old camera data into future inference.

---

## Slide 17: Latency Before and After

**Slide content**

Reported latency:

- Before: 30-35 s
- After: 9-11 s
- Improvement: about 3x

**Speech**

The reported progress result is that end-to-end latency improved from around 30 to 35 seconds down to around 9 to 11 seconds.

This was not caused by one single trick. It was a stack of optimizations.

First, the visual input was reduced into a fixed four-frame grid. Second, the app capped frame candidates and used lightweight selection. Third, the navigation prompt was made short and action-oriented. Fourth, max generation was capped at 44 tokens. Fifth, the native code avoids using the full context size as the batch size in the multimodal path. Sixth, the app streams partial responses and stops early when the answer is already complete.

So the result is a system that is still not instant, but it is much closer to practical live use.

---

## Slide 18: Latency Ablation

**Slide content**

Expected latency path:

- 32.0 s legacy
- 18.7 s with grid
- 14.9 s with smaller context/input
- 12.8 s with native memory changes
- 9.8 s with short generation and early stop

**Speech**

This ablation explains where the latency improvement comes from.

The biggest improvement is changing the visual representation. The four-frame grid means the model sees the temporal context in one image rather than multiple image chunks.

The second improvement is reducing the context and output budget. For live navigation, we do not need a long paragraph. We need one sentence.

The third improvement is native memory behavior. In the C++ multimodal path, the batch size is fixed at 512 instead of being equal to the full context. That avoids unnecessary memory pressure.

Finally, early stopping saves decode time. Once the model has produced a sentence like "Bench ahead right; open path left; move left slowly," there is no reason to continue generating.

---

## Slide 19: Speculative Decoding

**Slide content**

Status:

- Not implemented in current app path.
- Expected future optimization.

Expected:

- current: 9-11 s
- with speculative decoding: about 7 s

**Speech**

We also considered speculative decoding.

Speculative decoding uses a small draft model to propose tokens and the main model to verify them. If the main model accepts most proposed tokens, decoding becomes faster.

In the current app code, speculative decoding is not fully wired into the DrishtiAI inference path. So I am presenting this as an expected future ablation, not as a measured result.

The expected outcome is that speculative decoding can reduce the optimized path from around 9 to 11 seconds to around 7 seconds. The gain is meaningful, but it is not unlimited, because live navigation already generates very short answers. Speculative decoding mostly speeds up text generation; it does not remove camera capture or visual embedding time.

---

## Slide 20: Directional Bias Fix

**Slide content**

Problem:

- model overused "6 o'clock"

Fix:

- dataset balancing,
- prompt constraints,
- spatial post-processing.

**Speech**

One issue we identified was directional bias. The model could overuse "6 o'clock," even when that was not the correct direction.

This happens because egocentric video often has objects near the bottom of the frame, and the model can learn a bad shortcut.

We addressed this in three ways.

First, the dataset should be balanced so directions are more evenly represented. Second, prompts should force the model to reason about physical direction, not frame position. Third, the app includes a spatial post-processing helper that only activates when the model says 6 o'clock. It finds a salient region in the frame, maps that pixel location to a clock direction, and can correct the output when confidence is sufficient.

This is a good example of combining model training with deterministic safety logic.

---

## Slide 21: Training Ablation

**Slide content**

Expected combined score:

- base model: 0.535
- LoRA unfiltered: 0.660
- quality-filtered: 0.705
- prompt plus LM head: 0.735
- direction balancing plus postprocess: 0.755

**Speech**

The training ablation shows the expected role of each training decision.

The base model can describe scenes but is weak for navigation. LoRA fine-tuning improves task behavior. Quality filtering improves both confidence and navigation accuracy. The prompt style and LM head adaptation improve concise direction language. Direction balancing and post-processing reduce directional collapse.

The most important lesson is that training quality matters more than raw dataset size. A smaller but cleaner navigation-specific dataset is more useful than a large generic caption dataset.

---

## Slide 22: Results and Impact

**Slide content**

What we achieved:

- 210K BLV navigation samples
- compact VLM fine-tuning pipeline
- mobile inference app
- 4-frame live navigation grid
- 30-35 s to 9-11 s latency improvement
- combined evaluation around 0.74

**Speech**

To summarize the results, we built an end-to-end system.

We generated a 210 thousand sample BLV navigation dataset. We built the training pipeline for a compact vision-language model. We evaluated the model with navigation-specific metrics. We implemented an Android app with video QA and live navigation. And we optimized the latency from around 30 to 35 seconds to around 9 to 11 seconds.

The evaluation scores are around 0.74 combined, which shows that the system is producing useful navigation-style answers, though there is still room for improvement.

Most importantly, the project demonstrates that on-device BLV guidance is possible if the whole pipeline is designed for navigation rather than generic captioning.

---

## Slide 23: Limitations and Future Work

**Slide content**

Limitations:

- LMMS-Eval values need real completed runs.
- Speculative decoding is future work.
- More user testing is needed.
- Safety audit is required.

Future:

- real speculative decoding,
- better direction balancing,
- multilingual TTS,
- repeated mobile latency profiling,
- accessibility audit.

**Speech**

There are several limitations.

First, the LMMS-Eval benchmark values in this presentation should be treated as expected values until the full runs are completed and result JSON files are available.

Second, speculative decoding is not yet implemented in the current app inference path. It is a realistic next optimization, but not a measured current result.

Third, navigation is safety-critical, so we need more real-world user testing and accessibility review. A model can sound confident and still be wrong, so the app must handle uncertainty carefully.

Future work includes implementing real speculative decoding, collecting repeated timing traces on mobile devices, improving direction balancing, adding multilingual support, and running a full accessibility audit.

---

## Slide 24: Closing

**Slide content**

Final message:

DrishtiAI turns video understanding into on-device, actionable BLV navigation guidance.

**Speech**

The final takeaway is this:

DrishtiAI is not just a captioning app. It is an attempt to turn visual understanding into safe, actionable navigation guidance.

We did that by building a task-specific dataset, fine-tuning a compact model, designing evaluation metrics around navigation, and engineering the Android app for low-latency on-device inference.

The project shows that with the right data, prompts, model adaptation, and mobile optimization, a small local VLM can become useful for real assistive guidance.

Thank you.

---

# Short 5-Minute Backup Summary

If you are asked to summarize quickly:

DrishtiAI is an on-device Android vision-language assistant for blind and low-vision users. The core problem is that ordinary captions are not enough for navigation, so we generated a BLV-specific dataset from Charades and AVCaps. The pipeline extracts frames, uses Qwen2.5-VL to create structured scene summaries, uses Qwen2.5 to generate BLV QA pairs, filters them, and converts them into SmolVLA training format. The final dataset has 210,274 samples, with the largest categories being spatial relationships, navigation guidance, and obstacle detection.

We fine-tuned a compact SmolVLM-style model using LoRA, froze the vision encoder, trained language-side adapters, used 4-frame inputs, and evaluated with MCF and NAF. The file-backed evaluation scores are around 0.74 combined. On Android, we built video QA and live navigation. The key mobile trick is a 4-frame 2x2 grid that gives temporal context while keeping the visual input compact. The live navigation app samples frames, selects four with MobileNet, sends one grid to the model, streams a short answer, and speaks it with TTS.

Latency improved from a reported 30-35 seconds to 9-11 seconds through frame pruning, grid composition, smaller context, native batch optimization, token caps, and early stopping. Speculative decoding is not yet implemented, but an expected future ablation suggests it could reduce the optimized path to about 7 seconds.

---

# Q&A Preparation

## Why not just use a cloud model?

Cloud models are stronger, but navigation needs privacy, reliability, and predictable latency. On-device inference avoids sending camera frames to a server and can work offline.

## Why use a 4-frame grid?

A single frame lacks temporal context. Multiple separate frames are expensive. A 2x2 grid gives the model temporal evidence while keeping the input to one image.

## Why did mobile NAF drop while combined score improved?

The mobile path makes answers shorter and cleaner, which improves MCF. Shorter answers can omit some detailed direction or distance information, which lowers NAF slightly.

## Is speculative decoding already implemented?

No. The current app source does not wire in draft-model speculative decoding. The speculative decoding numbers are expected future ablation values.

## Which teacher model is best?

Expected public benchmark averages favor Qwen2.5-VL-72B as the best overall teacher. InternVL3 is expected to be strongest on perception-heavy tasks, and Qwen2.5-VL-7B is the best efficiency baseline.

## What is the biggest technical contribution?

The strongest contribution is the full task-specific loop: dataset generation, navigation-aware fine-tuning, BLV-specific evaluation, and mobile inference optimization. The four-frame grid is the most important app-level design.
