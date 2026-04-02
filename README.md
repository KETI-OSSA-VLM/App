# 적응형 KV 캐시 재사용 기반 Android 온디바이스 영상 VLM 시스템

> 최근 모바일 환경에서 비전 언어 모델(VLM)의 활용 요구가 증가하고 있으나, 기존 온디바이스 VLM 시스템은 주로 단일 정지 이미지만을 지원하며, 클라우드 기반 영상 VLM 서비스는 네트워크 지연 및 프라이버시 침해 한계가 존재한다. 본 연구에서는 영상을 입력으로 온디바이스에서 직접 VLM을 구동하는 안드로이드 네이티브 시스템을 설계 및 구현하였다. 본 시스템은 프레임 간 시각적 차이(MAD)를 분석하여 전체 모델 추론, KV 캐시 재사용, 결과 텍스트 재사용을 동적으로 선택하는 3-Tier 적응형 구조를 채택하였다. 이를 통해 시각적 변화가 적은 구간에서 불필요한 이미지 인코딩을 생략하고 텍스트 생성 속도를 효과적으로 개선하였다. 안드로이드 디바이스에서의 성능 평가 결과, 제안하는 시스템은 기존의 전체 추론 방식 대비 평균 지연 시간을 93% 감소시켰으며, 동일 시간 내 8배 이상의 프레임을 처리함으로써 모바일 환경에서의 효율적이고 실용적인 영상 VLM 구동 가능성을 입증하였다.

<!-- 데모 GIF -->

---

## 개요

모바일 기기에서 VLM을 영상에 적용하면 프레임당 ~6초의 추론 시간이 소요되어 실용적인 사용이 어렵다.

본 프로젝트는 **3단계 적응형 추론 시스템**을 제안한다. 프레임 간 시각적 변화량을 기반으로 추론 강도를 동적으로 선택하여, 매 프레임 전체 추론 대비 **평균 93% 레이턴시 감소**와 **8배 많은 프레임 처리**를 달성한다.

### 핵심 실험 결과

| 방법 | 평균 레이턴시 / 프레임 | 처리 프레임 수 |
|------|---------------------|--------------|
| Baseline (항상 T2) | ~6,035ms | 37 |
| **제안 방법 (Adaptive)** | **~430ms** | **296** |

> 동일한 영상으로 테스트. 기기: Android Snapdragon (CPU 전용 추론)

---

## 동작 원리

입력 프레임을 직전 프레임과 **32×32 MAD(Mean Absolute Difference)** 로 비교하여 3단계 중 하나를 선택한다.

```
┌─────────────────────────────────────────────────────────┐
│  프레임 N 입력                                            │
│       ↓                                                  │
│  직전 프레임과 MAD diff 계산                               │
│       ↓                                                  │
│  diff < 1%      →  T0: 캐시 결과 재사용       (~15ms)    │
│  1% ≤ diff < 5% →  T1: KV 캐시 재사용        (~850ms)   │
│  diff ≥ 5%      →  T2: 전체 추론             (~6,400ms)  │
└─────────────────────────────────────────────────────────┘
```

| Tier | 조건 | 동작 | 레이턴시 |
|------|------|------|---------|
| **T0** | diff < 1% | 이전 결과 캐시 반환 | ~15ms |
| **T1** | 1% ≤ diff < 5% | 이미지 인코딩 생략, KV 캐시로 텍스트 재생성 | ~850ms |
| **T2** | diff ≥ 5% | 전체 추론 (이미지 인코딩 + 텍스트 생성) | ~6,400ms |

### T1: KV 캐시 재사용

T1은 마지막 T2 추론의 이미지 KV 캐시를 재사용하여 비용이 큰 비전 인코딩 단계를 생략한다. 텍스트 토큰만 재생성하므로 T2 대비 약 7배 빠르다.

```
T2:  [이미지 인코딩] → [Prefill] → [Decode]   ~6,400ms
T1:                   [KV 재사용] → [Decode]    ~850ms
T0:  ← 캐시 결과 반환 →                          ~15ms
```

### Tier 분포 (Adaptive 모드)

<!-- Tier 분포 GIF -->

| Tier | 비율 |
|------|------|
| T0 | 73% |
| T1 | 24% |
| T2 | 3% |

---

## 시스템 구조

```
MainActivity
    │
    ├── VideoFileFrameExtractor       (100ms 간격, CONFLATED 채널)
    │       ↓ Bitmap
    ├── AdaptiveVlmRunner
    │       ├── FrameDiffAnalyzer     (32×32 MAD → T0 / T1 / T2 판정)
    │       └── LlamaCppEngine
    │               ├── generate()        → T2 전체 추론 (384px JPEG)
    │               └── generateOnly()    → T1 KV 캐시 재사용
    │                       ↕ JNI
    └── llama_bridge.cpp
            ├── generate()            (이미지 인코딩 + prefill + decode)
            └── generateOnly()        (seq_rm → prime decode → decode)
```

**네이티브 레이어** (`llama_bridge.cpp`):
- `generate()`: [llama.cpp](https://github.com/ggerganov/llama.cpp) + MTMD 기반 멀티모달 전체 추론
- `generateOnly()`: `llama_memory_seq_rm`으로 KV 캐시를 prefill 직후 상태로 복원 후 텍스트 재생성

---

## 모델

**[SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)** (GGUF, Q4 양자화)
- 500M 파라미터 — 온디바이스 추론에 적합한 경량 VLM
- 필요 파일: `smolvlm2-500m.gguf` + `mmproj.gguf`
- 추론: CPU 전용 (llama.cpp, GPU 미사용)

---

## 시작하기

### 사전 요구사항

- Android Studio (Hedgehog 이상)
- Android 기기 (API 28+, arm64-v8a)
- WSL (Ubuntu) + [Android NDK r27c](https://developer.android.com/ndk/downloads)

### 1. 클론

```bash
git clone --recurse-submodules https://github.com/KETI-OSSA-VLM/App.git
cd App
```

### 2. 모델 파일 다운로드

아래 파일을 기기의 `/sdcard/Android/data/com.example.genionputtest/files/` 에 넣는다.

```
smolvlm2-500m.gguf
mmproj-smolvlm2-500m.gguf
```

### 3. 네이티브 라이브러리 빌드

```bash
# WSL (Ubuntu) 에서 실행
wsl -d Ubuntu bash /mnt/c/android/accv/build_accv_llama.sh
```

`llama_bridge.cpp`를 NDK r27c로 컴파일하여 `libaccv_llama.so`를 생성한다.

### 4. 실행

Android Studio에서 프로젝트를 열고 **Run** 버튼을 누른다.

---

## 평가

앱에서 **Eval 모드**를 활성화하면 프레임별 tier, 레이턴시, diff score가 CSV로 저장된다.

```
frame,timestamp_ms,tier,latency_ms,diff_score
0,1775098927886,TWO,6128.0,1.0000
1,1775098928736,ONE,837.0,0.0202
2,1775098929770,ONE,1022.0,0.0121
3,1775098929797,ZERO,10.0,0.0080
...
```

**Baseline 모드**를 활성화하면 매 프레임 T2로 강제 실행되어 비교 실험이 가능하다.

---

## 관련 연구

| 연구 | 학회/출처 | 핵심 내용 |
|------|---------|---------|
| [HERMES](https://arxiv.org/abs/2601.14724) | ICLR 2026 | 계층적 KV 캐시로 스트리밍 영상 처리 (서버 GPU) |
| [LiveVLM](https://arxiv.org/abs/2505.15269) | arXiv 2025 | 스트리밍 KV eviction, 44× 속도 향상 (서버 GPU) |
| [StreamingVLM](https://arxiv.org/html/2510.09608v1) | MIT Han Lab | Compact KV window, 8 FPS (서버 GPU) |
| [VideoLLM-online](https://github.com/showlab/videollm-online) | CVPR 2024 | 비동기 스트리밍 파이프라인 (서버 GPU) |
| [OFFGRID](https://github.com/alichherawalla/off-grid-mobile-ai) | — | 온디바이스 VLM 앱 (React Native, llama.cpp) |

> 위 연구들은 모두 서버 GPU를 대상으로 한다. 본 프로젝트는 llama.cpp를 통해 Android에서 직접 KV 캐시 재사용 기반 영상 추론을 구현한 최초의 모바일 시스템이다.

---

## 한계점

- **T1은 현재 프레임이 아닌 마지막 T2 프레임의 이미지 컨텍스트를 기반으로 텍스트를 생성**한다. 이미지 KV 캐시의 부분 업데이트(모델 레벨)는 향후 연구 방향이다.
- **T2 레이턴시 (~6.4s)** 가 병목이다. GPU 가속 또는 경량 비전 인코더로 개선 가능하다.
- **T2 진행 중 취소** 시 JNI 호출 완료까지 최대 6.4초 지연이 발생한다.

---

## 인용

```bibtex
@article{accv2026adaptive,
  title   = {Android On-Device Video VLM System Based on Adaptive KV Cache Reuse},
  author  = {Sooin Jung},
  journal = {IEMEK},
  year    = {2026}
}
```

---

## 감사의 글

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 추론 엔진 및 MTMD 멀티모달 확장
- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) — 기반 비전 언어 모델
