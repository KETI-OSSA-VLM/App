# AGENTS.md

## Project

- 이름: `accv`
- 설명: Android 보드 환경에서 TFLite와 NNAPI 기반 비전 추론을 검증하고, 최종적으로 StreamingVLM 포팅 가능성을 단계적으로 확인하는 프로젝트다.
- 현재 목표: 단일 TFLite 분류 모델에서 실제 이미지 입력 기반 추론 파이프라인을 완성하고, 이후 vision encoder 계열 모델 실험으로 확장할 수 있는 베이스 앱을 만든다.
- 비목표: 외부 공유용 공식 문서 작성, 과도한 위키화, 세부 구현 문서를 초기에 과도하게 분리하는 일

## Source Of Truth

- 세션 시작 시 먼저 이 문서를 읽는다.
- 다음으로 [`memory/current.md`](/C:/android/accv/memory/current.md)를 읽는다.
- 상세 작업 이력은 [`memory/daily/`](/C:/android/accv/memory/daily)에서 확인한다.
- 중요한 결정은 [`memory/decisions/`](/C:/android/accv/memory/decisions)에서 확인한다.
- 아직 정리되지 않은 메모는 [`memory/scratchpad.md`](/C:/android/accv/memory/scratchpad.md)를 확인한다.

## Working Rules

- 작업 시작 전 현재 목표와 열린 이슈를 `memory/current.md`에서 확인한다.
- 작업 종료 시 `memory/current.md`를 최신 상태로 갱신한다.
- 당일 작업 내용은 `memory/daily/YYYY-MM-DD.md`에 기록한다.
- 큰 방향 변경이나 되돌아볼 가치가 있는 판단은 `memory/decisions/`에 별도 문서로 남긴다.
- 확정되지 않은 아이디어나 임시 TODO는 `memory/scratchpad.md`에 적고, 필요 시 정식 문서로 옮긴다.

## Current Priorities

- 실제 이미지 입력 기반 MobileNet 분류 앱을 완성한다.
- 전처리, 추론, 후처리, latency breakdown이 앱 안에서 정상 동작하는지 검증한다.
- 이후 vision encoder 단독 실행 단계로 넘어갈 수 있는 재사용 가능한 추론 베이스를 만든다.

## Constraints

- 현재 저장소 루트에는 Android/Gradle 프로젝트 파일이 있지만, 이 폴더 자체는 아직 Git 저장소로 초기화되어 있지 않다.
- 문서 구조는 유지 비용이 낮아야 하며, 다음 세션 시작 시 1분 안에 이해 가능해야 한다.
- `memory/current.md`는 짧은 메모가 아니라 상세한 현재 상태 문서로 유지한다.
- StreamingVLM으로 바로 가면 실패 원인을 모델, 비전 인코더, 디코더, NNAPI delegate 문제로 분리하기 어렵다.

## Update Routine

1. 세션 시작: `AGENTS.md` 확인
2. 현재 상태 확인: `memory/current.md` 확인
3. 작업 수행 중 메모: 필요 시 `memory/scratchpad.md` 기록
4. 세션 종료: `memory/daily/YYYY-MM-DD.md`와 `memory/current.md` 갱신
