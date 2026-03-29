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
| **Sprint 2** | 3/18 ~ 3/19 | 본 실험 (양자화+평가), C_v3 알고리즘 개선, KoBEST+kmmlu 결론 도출 | ✅ 완료 |
| **Sprint 3** | 3/20 ~ 4/13 | 결론 강화 + 언어 일반화 검증 (★ 4/17 중간 보고서) | ✅ 실험 완료 (2026-03-26) |
| **Sprint 4** | 4/14 ~ 6/4 | 논문 작성 (★ 6/5 쇼츠 제작물, 6/12 최종 마감) | ⬜ 대기 |

---

## Sprint 2 완료 항목 (2026-03-19 최종)

- ✅ GPTQ 4-bit 양자화 파이프라인 구축 (A/B/C/C_v2/C_v3/C_v4)
- ✅ lm-evaluation-harness KoBEST + kmmlu 연동
- ✅ FP16 베이스라인 측정 (kobest avg 0.6523)
- ✅ C_v3 알고리즘 재설계 (비한국어 subword 제거) → **best 조건 확정**
- ✅ KoBEST: C_v3(97.4%) > B(94.7%) > C(92.9%) > C_v4(92.1%) > A(91.7%) > C_v2(88.6%)
- ✅ kmmlu 교차 검증: C(0.3752) > C_v3(0.3747) > C_v4/B(0.3731~) > A(0.3677)
- ✅ C_v3 재현성 확인: Run1=0.6354, Run2=0.6356 (차이 0.0002)
- ✅ C_v4 길이 효과 분리: 어절 수 증가는 성능 개선 없음 → C_v3 우위는 비한국어 제거 덕분
- ✅ Activation 분포 분석: C_v3가 channel_cv/mean_std 최고 — 다양한 activation 유도 확인
- ✅ 메커니즘 규명: 비한국어 subword 과다(846개→570개)가 GPTQ 왜곡 원인

## Sprint 2 → Sprint 3 이월 항목

- ❌ 다중 런 95% CI — 2회 재현으로 방향성 확인했으나 통계 검정 미실시
- ❌ Hessian 근사 오차와 activation 다양성 직접 연결 — 인과 관계 미검증
- ❌ 단일 모델 검증만 — SOLAR 외 모델 일반화 미착수

---

## Sprint 3 목표 (3/20 ~ 4/13, ★ 4/17 중간 보고서)

> **논문 서사 구조:**
> 1. 한국어 모델 여러 개에서 C_v3 일관 우위 → "한국어 PTQ의 일반 원리"
> 2. 타 언어(Chinese) 모델로 확장 → "언어 정렬 + 형태소 다양성 = 언어 불문 원리"

### 3-A: 결론 강화
1. C_v3 다중 런 (3회 이상) → 95% CI 산출 — B vs C_v3 차이(0.018) 통계 유의성 확인
2. Hessian 근사 오차 측정 — activation 다양성과 직접 연결 (선택)

### 3-B: 언어 정렬 원리 일반화 ← Sprint 3 핵심 (수정됨 2026-03-19)

**Sprint 2 다모델 검증으로 가설 정제됨:**

단순 "한국어 calibration 만능"이 아니라 **"사전학습 주 언어 = calibration 언어"** 원리로 재정립.

| 모델 | 사전학습 | A | B | C_v3 | 최우수 |
|------|---------|---|---|------|--------|
| SOLAR-10.7B | 한국어 | 0.5981 | 0.6176 | **0.6356** | C_v3 ✅ |
| Llama3-Ko-8B | 영어(LLaMA-3) | **0.5758** | 0.5608 | 0.5650 | A |
| EEVE-10B | 한국어(SOLAR) | 0.7172 | **0.7595** | 0.7514 | B |

**Sprint 3 핵심 실험:**

| 실험 | 모델 | 사전학습 주 언어 | 검증 포인트 |
|------|------|----------------|------------|
| ① | **Qwen2-7B** | 중국어 | 중국어 diverse calibration > 한국어 > 영어? |
| ② | **EXAONE-3.5** (호환 환경 구성 후) | 한국어 전용 | SOLAR처럼 C_v3 > B? |

- Qwen2: jieba로 중국어 형태소 다양성 calibration 구현 → C-Eval / CMMLU 평가
- 성공 시 논문 주장: **"calibration 언어를 모델 사전학습 주 언어에 맞출수록 GPTQ 품질 보존"**

### 3-D: Sprint 3 완료 결과 (2026-03-26)

**S3-1: Qwen2-7B 실험**
- KoBEST: C_v3(0.6375) > A(0.6238) > C_zh(0.6063) — 가설(C_zh 최선) 기각
- C-Eval (중국어 벤치마크): C_v3(0.7694) > A(0.7617) > C_zh(0.7584) — 중국어 모델에서도 C_zh가 최하
- 해석: "사전학습 언어 정렬"보다 형태소 다양성 알고리즘 자체 효과 또는 C_zh avg_SFS 낮음이 원인

**S3-2: EXAONE-3.5-7.8B 실험**
- FP16=0.7437, A=0.6645(89.3%), B=0.7196(96.8%), C_v3=0.7164(96.3%)
- 패턴: B≈C_v3 >> A (SOLAR과 달리 C_v3 > B 불명확) → tokenizer 의존성 재확인

**S3-3: EEVE C_v3_eeve 실험**
- C_v3_eeve=0.7551 vs C_v3=0.7514 (+0.0037) vs B=0.7595 (여전히 최선)
- tokenizer 정렬 소폭 효과 있으나 B 우위 뒤집기 실패

**S3-Phase2: SOLAR 양자화 변형 실험 (GPU0)**
- g64: C_v3=0.6468(+0.011), A=0.6174(+0.019) — g64가 품질 향상, C_v3-A 격차는 0.037→0.029 축소
- SmoothScale+GPTQ: 0.48대 (❌ alpha=0.5 과도, Hessian 붕괴)
- desc_act=False: C_v3=0.6029 (desc_act=True 대비 −0.033, calibration 우위는 유지)

### 3-C: 중간 보고서 작성 (4/17 마감)
- **내용 범위**: 읽은 논문 정리 + 연구 방향 선택 이유 + 연구 아이디어 제안
- 실험 결과는 필수 아님 (있으면 플러스)
- 핵심: PTQ calibration 데이터 언어 정렬 문제를 왜 형태소 다양성 관점에서 접근했는지 서술
- Sprint 2 결과(SOLAR C_v3 97.4% 보존) + 다모델 발견(언어 정렬 원리 단서)을 예비 결과로 포함

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

