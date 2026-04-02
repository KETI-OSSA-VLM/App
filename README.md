# Adaptive Video Inference for On-Device VLM on Android

> Real-time streaming Vision Language Model inference on Android via frame difference-guided adaptive KV cache reuse.

<!-- Demo GIF here -->

---


## Overview

Running VLMs on mobile devices is slow. Full inference per frame (~6s) makes real-time video understanding impractical.

This project proposes a **3-tier adaptive inference system** that dynamically selects the inference mode per frame based on visual change — achieving **93% average latency reduction** compared to full inference on every frame, with **8× more frames analyzed** in the same time.

### Key Results

| Method | Avg Latency / Frame | Frames Processed |
|--------|-------------------|-----------------|
| Baseline (always T2) | ~6,035ms | 37 |
| **Ours (Adaptive)** | **~430ms** | **296** |

> Tested on the same video. Device: Android with Snapdragon (CPU-only inference).

---


## How It Works

Each incoming frame is compared to the previous frame using **32×32 Mean Absolute Difference (MAD)**. Based on the diff score, one of three inference tiers is selected:

```
┌─────────────────────────────────────────────────────────┐
│  Frame N arrives                                         │
│       ↓                                                  │
│  MAD diff vs. Frame N-1                                  │
│       ↓                                                  │
│  diff < 1%  →  T0: Reuse cached result       (~15ms)    │
│  1% ≤ diff < 5%  →  T1: KV cache reuse      (~850ms)   │
│  diff ≥ 5%  →  T2: Full inference           (~6,400ms)  │
└─────────────────────────────────────────────────────────┘
```

| Tier | Condition | Action | Latency |
|------|-----------|--------|---------|
| **T0** | diff < 1% | Return cached text | ~15ms |
| **T1** | 1% ≤ diff < 5% | Skip image encoding, regenerate text from KV cache | ~850ms |
| **T2** | diff ≥ 5% | Full inference (image encode + text generation) | ~6,400ms |


### T1: KV Cache Reuse

T1 skips the expensive vision encoding step by reusing the image KV cache from the last T2 inference. Only text tokens are regenerated — approximately 7× faster than T2.

```
T2:  [Image Encode] → [Prefill] → [Decode]   ~6,400ms
T1:                   [KV Reuse] → [Decode]    ~850ms
T0:  ← cached result →                          ~15ms
```


### Tier Distribution (Adaptive Mode)

<!-- Tier bar visualization GIF here -->

| Tier | Share |
|------|-------|
| T0 | 73% |
| T1 | 24% |
| T2 | 3% |

---


## Architecture

```
MainActivity
    │
    ├── VideoFileFrameExtractor       (100ms frame interval, CONFLATED channel)
    │       ↓ Bitmap
    ├── AdaptiveVlmRunner
    │       ├── FrameDiffAnalyzer     (32×32 MAD → T0 / T1 / T2)
    │       └── LlamaCppEngine
    │               ├── generate()        → T2 full inference   (384px JPEG)
    │               └── generateOnly()    → T1 KV cache reuse
    │                       ↕ JNI
    └── llama_bridge.cpp
            ├── generate()            (image encode + prefill + decode)
            └── generateOnly()        (seq_rm → prime decode → decode)
```

**Native layer** (`llama_bridge.cpp`):
- `generate()`: full multimodal inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) + MTMD
- `generateOnly()`: restores KV cache to post-prefill state via `llama_memory_seq_rm`, then regenerates text

---


## Model

**[SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)** (GGUF, Q4 quantized)
- 500M parameters — practical for on-device inference
- Requires: `smolvlm2-500m.gguf` + `mmproj.gguf`
- Inference: CPU-only (llama.cpp, no GPU backend)

---


## Getting Started


### Prerequisites

- Android Studio (Hedgehog or later)
- Android device with API 28+ (arm64-v8a)
- WSL (Ubuntu) + [Android NDK r27c](https://developer.android.com/ndk/downloads)


### 1. Clone

```bash
git clone --recurse-submodules https://github.com/KETI-OSSA-VLM/App.git
cd accv
```


### 2. Download Models

Place the following files in `/sdcard/Android/data/com.example.genionputtest/files/`:

```
smolvlm2-500m.gguf
mmproj-smolvlm2-500m.gguf
```


### 3. Build Native Library

```bash
# In WSL (Ubuntu)
wsl -d Ubuntu bash /mnt/c/android/accv/build_accv_llama.sh
```

This compiles `llama_bridge.cpp` into `libaccv_llama.so` using NDK r27c.


### 4. Run

Open the project in Android Studio and press **Run**.

---


## Evaluation

Enable **Eval Mode** in the app to record per-frame tier, latency, and diff score as CSV.

```
frame,timestamp_ms,tier,latency_ms,diff_score
0,1775098927886,TWO,6128.0,1.0000
1,1775098928736,ONE,837.0,0.0202
2,1775098929770,ONE,1022.0,0.0121
3,1775098929797,ZERO,10.0,0.0080
...
```

Enable **Baseline Mode** to force T2 on every frame for comparison.

---


## Related Work

| Work | Venue | Key Idea |
|------|-------|----------|
| [HERMES](https://arxiv.org/abs/2601.14724) | ICLR 2026 | Hierarchical KV cache for streaming video (server GPU) |
| [LiveVLM](https://arxiv.org/abs/2505.15269) | arXiv 2025 | Streaming KV eviction, 44× speedup (server GPU) |
| [StreamingVLM](https://arxiv.org/html/2510.09608v1) | MIT Han Lab | Compact KV window, 8 FPS (server GPU) |
| [VideoLLM-online](https://github.com/showlab/videollm-online) | CVPR 2024 | Async streaming pipeline (server GPU) |
| [OFFGRID](https://github.com/alichherawalla/off-grid-mobile-ai) | — | On-device VLM app (React Native, llama.cpp) |

> Unlike the above works targeting server GPUs, this project implements KV cache reuse for streaming video inference directly on Android via llama.cpp — to the best of our knowledge, the first such implementation on mobile.

---


## Limitations

- **T1 describes the previous frame's scene**, not the current one — image encoding is skipped entirely. Partial KV cache update (model-level) is a direction for future work.
- **T2 latency (~6.4s)** is the bottleneck — determined by SmolVLM2-500M on CPU. GPU acceleration or a lighter vision encoder would reduce this.
- **Cancel during T2** waits for the JNI call to complete (~6.4s max delay).

---


## Citation

If you find this work useful, please cite:

```bibtex
@article{accv2026adaptive,
  title   = {Adaptive KV Cache Reuse for Real-Time Vision Language Model Inference on Android},
  author  = {Sooin Jung},
  journal = {IEMEK},
  year    = {2026}
}
```

---


## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — inference engine and MTMD multimodal extension
- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) — base vision language model
