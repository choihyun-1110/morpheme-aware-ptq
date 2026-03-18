# Sprint Overview

## 프로젝트: Morphology-Aware Calibration for PTQ of Korean LLMs

> **궁극적 비전:** 언어 정렬 + 형태소 다양성 최적화 calibration이 GPTQ 양자화 품질을 일반적으로 보존한다는 원리를 실증하고, 한국어에서 시작해 다국어로 확장한다.
> **실무 원칙:** 생각 먼저, 코드 나중. 모든 결정은 백로그 문서로 남기고 컨펌 후 구현한다.

---

## 스프린트 구성

| Sprint | 기간 | 목표 | 상태 |
|--------|------|------|------|
| **Sprint 0** | 3/12 ~ 3/16 | 환경 검증 + 실험 가능성 파일럿 | ✅ 완료 |
| **Sprint 1** | 3/17 ~ 3/18 | 대상 모델(SOLAR) 분석 및 Calibration Set 알고리즘 (A/B/C) | ✅ 완료 |
| **Sprint 2** | 3/18 ~ 3/23 | 본 실험 (양자화+평가), C_v3 알고리즘 개선, KoBEST 결론 도출 | 🔵 마무리 중 |
| **Sprint 3** | 3/24 ~ 4/13 | 결론 강화 + 언어 일반화 검증 (★ 4/17 중간 보고서) | ⬜ 대기 |
| **Sprint 4** | 4/14 ~ 6/4 | 논문 작성 (★ 6/5 쇼츠 제작물, 6/12 최종 마감) | ⬜ 대기 |

---

## Sprint 2 완료 항목 (2026-03-18 기준)

- ✅ GPTQ 4-bit 양자화 파이프라인 구축 (A/B/C/C_v2/C_v3)
- ✅ lm-evaluation-harness KoBEST 연동
- ✅ FP16 베이스라인 측정 (kobest avg 0.6523)
- ✅ C_v3 알고리즘 재설계 (비한국어 subword 제거) → **best 조건 확정**
- ✅ KoBEST 최종 순위: C_v3(97.4%) > B(94.7%) > C(92.9%) > A(91.7%) > C_v2(88.6%)
- ✅ 메커니즘 규명: 비한국어 subword 과다(846개→570개)가 GPTQ 왜곡 원인

## Sprint 2 미완료 항목 (→ Sprint 3 이월)

- 🔄 C_v3 재현성 확인 (3회 이상 다중 런)
- 🔄 kmmlu 교차 검증 (A/B/C/C_v3 전체)
- 🔄 C_v4 (어절 수 B 수준 15개) → 길이 효과 분리
- ❌ S2-4: Activation 분석 — calibration subword 분포 → activation 다양성 연결 미착수

---

## Sprint 3 목표 (3/24 ~ 4/13, ★ 4/17 중간 보고서)

### 3-A: Sprint 2 마무리
1. C_v3 다중 런 → 95% CI / C_v4 결과 분석
2. kmmlu 전체 조건 완료 → 교차 검증
3. Activation 분석 (S2-4 이월) — 핵심 가설 링크 검증

### 3-B: 언어 일반화 검증 ← 논문 핵심 기여
1. **Qwen2-7B (Chinese-dominant)** — Chinese diverse vs Korean vs English calibration
2. LLaMA-3 (English-dominant) — 역방향 검증
3. 결과: "언어 정렬 + 다양성 최적화 = 일반 원리" 입증

### 3-C: 중간 보고서 작성 (4/17 마감)
- **내용 범위**: 읽은 논문 정리 + 연구 방향을 선택한 이유 + 연구 아이디어 제안
- 실험 결과는 필수 아님 (있으면 플러스)
- 핵심: PTQ calibration 데이터 선택 문제를 왜 한국어 형태소 관점에서 접근했는지 서술

---

## 워크플로우

```
백로그 작성 (.md)
  → 사고 과정 토론/정리
  → 컨펌
  → 코드 구현
  → 결과 기록
```

## 디렉토리 구조

```
llm_project/
├── planning.md              # 전체 연구 기획서
├── pilot_quant.py           # Sprint 0 파일럿 양자화 스크립트
├── sprints/
│   ├── sprint_overview.md   # 이 파일 (전체 일정)
│   ├── sprint0/
│   │   ├── backlog.md
│   │   └── thoughts/
│   │       ├── 00_환경검증_어떤모델부터.md
│   │       ├── 01_논문기반_플랜재검토.md
│   │       ├── 02_양자화_라이브러리_선정.md
│   │       ├── 03_형태소분석기_선정.md
│   │       └── 04_평가프레임워크_선정.md
│   └── sprint1/
│       ├── backlog.md
│       └── thoughts/
│           ├── 01_EXAONE_아키텍처_스터디.md
│           ├── 02_대상모델_변경_업스테이지_Llama.md
│           ├── 03_캘리브레이션셋_알고리즘_설계.md
│           ├── 03_ref_다양성지표_출처.md
│           └── 04_나무위키_데이터셋_선정.md
├── src/                     # 코드 (S1-3에서 구현)
│   ├── build_calibration.py # 메인 CLI (조건 A/B/C/all)
│   ├── preprocess.py        # 나무위키 전처리 + 문장 분리
│   ├── diversity.py         # 다양성 점수 (UMR/TTR/SFS)
│   ├── selection.py         # Greedy Diversity Selection
│   ├── visualize.py         # 시각화 (alignment + 분포)
│   └── requirements.txt     # 의존성
├── results/                 # 실험 결과 (S1-4에서 생성)
└── docs/
    ├── server_setup_guide.md
    └── 학습필요_메모.md       # LLM Tokenizer, GPTQ 등 학습 항목
```

