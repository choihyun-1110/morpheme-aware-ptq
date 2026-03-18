"""
Greedy Diversity Selection 모듈
- 형태소 커버리지를 극대화하며 128문장 선별
- 너무 짧은 문구/표제어가 과도하게 선택되지 않도록 문장성 보정
"""
import re
from typing import List, Set, Tuple
from diversity import SentenceScore
import numpy as np

# coverage_bonus 계산에서 제외할 Kiwi 품사 태그 (비한국어)
_NON_KO_TAGS = {"SL", "SH", "SN", "SW"}  # 외국어, 한자, 숫자, 기호


def _ko_char_ratio(text: str) -> float:
    """텍스트에서 한국어(가-힣) 문자가 차지하는 비율."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    ko = sum(1 for c in chars if "\uAC00" <= c <= "\uD7A3")
    return ko / len(chars)


def _ko_morpheme_set(morpheme_set: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """SL/SH/SN/SW 태그를 제외한 한국어 형태소 집합만 반환."""
    return {m for m in morpheme_set if m[1] not in _NON_KO_TAGS}


def has_sentence_final_ending(candidate: SentenceScore) -> bool:
    """
    문장 종결형 여부를 간단한 휴리스틱으로 판정.

    - 문장부호(. ! ? …)로 끝나면 문장으로 간주
    - Kiwi 마지막 형태소 태그가 EF/SF 이면 문장 종결로 간주
    """
    text = candidate.text.strip()
    if text.endswith((".", "!", "?", "…")):
        return True

    if not candidate.morphemes:
        return False

    last_tag = candidate.morphemes[-1][1]
    if last_tag in {"EF", "SF"}:
        return True

    if len(candidate.morphemes) >= 2:
        penultimate_tag = candidate.morphemes[-2][1]
        if penultimate_tag == "EF" and last_tag == "SF":
            return True

    return False


def sentence_length_score(
    candidate: SentenceScore,
    min_eojeols: int = 5,
    target_eojeols: int = 12,
    min_subword_tokens: int = 24,
    target_subword_tokens: int = 64,
) -> float:
    """
    너무 짧은 문구를 피하기 위한 길이 점수.
    최소 기준 미만이면 0, 목표 길이에 도달하면 1.
    """

    def ramp(value: int, minimum: int, target: int) -> float:
        if value <= minimum:
            return 0.0
        if value >= target:
            return 1.0
        return (value - minimum) / (target - minimum)

    eojeol_score = ramp(candidate.n_eojeols, min_eojeols, target_eojeols)
    token_score = ramp(len(candidate.token_ids), min_subword_tokens, target_subword_tokens)
    return 0.5 * eojeol_score + 0.5 * token_score


def filter_sentence_like_candidates(
    candidates: List[SentenceScore],
    min_eojeols: int = 5,
    min_subword_tokens: int = 24,
    require_sentence_final: bool = True,
    min_ko_ratio: float = 0.0,
) -> List[SentenceScore]:
    """
    C 조건에서 제목/문구형 샘플이 과도하게 선택되는 것을 막기 위한 사전 필터.

    Args:
        min_ko_ratio: 0 이상이면 한국어 문자 비율이 이 값 미만인 문장을 제거.
                      0.0(기본값)이면 필터 비활성.
    """
    filtered = []
    for candidate in candidates:
        if candidate.n_eojeols < min_eojeols:
            continue
        if len(candidate.token_ids) < min_subword_tokens:
            continue
        if require_sentence_final and not has_sentence_final_ending(candidate):
            continue
        if min_ko_ratio > 0.0 and _ko_char_ratio(candidate.text) < min_ko_ratio:
            continue
        filtered.append(candidate)

    print(
        f"[선택] 문장형 후보 필터링: {len(candidates)} -> {len(filtered)}"
        f" (min_eojeols={min_eojeols}, min_subword_tokens={min_subword_tokens}, "
        f"require_sentence_final={require_sentence_final}, min_ko_ratio={min_ko_ratio})"
    )
    return filtered


def greedy_diversity_select(candidates: List[SentenceScore],
                             n: int = 128,
                             alpha: float = 0.3,
                             beta: float = 0.15,
                             gamma: float = 0.15,
                             min_eojeols: int = 5,
                             target_eojeols: int = 12,
                             min_subword_tokens: int = 24,
                             target_subword_tokens: int = 64,
                             ) -> List[SentenceScore]:
    """
    Greedy Diversity Selection 알고리즘.
    
    이미 선택된 문장의 형태소와의 중복을 페널티로 주어,
    전체 형태소 커버리지를 극대화하면서 다양성이 높은 문장을 선별한다.
    추가로 길이/문장 완결성 보너스를 줘서 짧은 문구 편향을 완화한다.
    
    Args:
        candidates: 다양성 점수가 산출된 SentenceScore 리스트
        n: 선택할 문장 수
        alpha: 커버리지 보너스 계수 (높을수록 새 형태소 우선)
        beta: 길이 보너스 계수
        gamma: 문장 종결 보너스 계수
    Returns:
        selected: 선별된 n개 SentenceScore 리스트
    """
    selected = []
    selected_indices = set()
    covered_morphemes: Set[Tuple[str, str]] = set()  # 한국어 형태소만 추적

    for step in range(n):
        best_score = float('-inf')
        best_idx = -1

        for i, cand in enumerate(candidates):
            if i in selected_indices:
                continue

            # 기본 다양성 점수
            base_score = cand.diversity_score

            # 신규 형태소 커버리지 보너스 — 한국어 형태소만 사용
            # (한자/외국어/숫자/기호는 tokenizer 분절 특성 때문에 coverage를 왜곡함)
            cand_ko = _ko_morpheme_set(cand.morpheme_set)
            new_morphemes = cand_ko - covered_morphemes
            coverage_bonus = len(new_morphemes) / (len(cand_ko) + 1e-8)

            # 너무 짧은 문구 편향을 줄이기 위한 길이 보너스
            length_bonus = sentence_length_score(
                cand,
                min_eojeols=min_eojeols,
                target_eojeols=target_eojeols,
                min_subword_tokens=min_subword_tokens,
                target_subword_tokens=target_subword_tokens,
            )

            # 종결 어미/문장부호를 가진 문장에 소폭 보너스
            sentence_bonus = 1.0 if has_sentence_final_ending(cand) else 0.0
            
            # 최종 선택 점수
            final_score = (
                base_score
                + alpha * coverage_bonus
                + beta * length_bonus
                + gamma * sentence_bonus
            )
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        
        if best_idx == -1:
            print(f"[선택] 경고: {step}번째에서 더 이상 선택 가능한 문장이 없습니다.")
            break
        
        selected.append(candidates[best_idx])
        selected_indices.add(best_idx)
        covered_morphemes |= _ko_morpheme_set(candidates[best_idx].morpheme_set)
        
        if (step + 1) % 32 == 0:
            print(f"[선택] {step+1}/{n} 문장 선택 완료, "
                  f"누적 형태소 커버리지: {len(covered_morphemes)}종")
    
    print(f"\n[선택] 최종 결과: {len(selected)}문장 선택")
    print(f"  - 총 고유 형태소 커버리지: {len(covered_morphemes)}종")
    print(f"  - 평균 다양성 점수: {np.mean([s.diversity_score for s in selected]):.3f}")
    
    return selected


def export_calibration_set(selected: List[SentenceScore],
                            model_name: str,
                            output_path: str,
                            params: dict = None):
    """
    선별된 문장을 JSON 파일로 저장.
    
    Args:
        selected: 선별된 SentenceScore 리스트
        model_name: 대상 모델 이름
        output_path: 출력 JSON 파일 경로
        params: 선택 파라미터 기록용 dict
    """
    import json
    from collections import Counter
    
    # 형태소 품사 분포 집계
    all_tags = []
    for s in selected:
        all_tags.extend([tag for _, tag in s.morphemes])
    tag_dist = dict(Counter(all_tags).most_common())
    
    # 전체 커버리지 집계
    all_morphemes = set()
    for s in selected:
        all_morphemes |= s.morpheme_set
    
    output = {
        "model": model_name,
        "condition": "C_morphology_diverse",
        "n_sentences": len(selected),
        "selection_params": params or {
            "weights": [1/3, 1/3, 1/3],
            "alpha": 0.3,
        },
        "sentences": [
            {
                "text": s.text,
                "diversity_score": round(s.diversity_score, 4),
                "umr": round(s.umr, 4),
                "ttr": round(s.ttr, 4),
                "sfs": round(s.sfs, 2),
                "n_morphemes": len(s.morphemes),
                "n_unique_morphemes": len(s.morpheme_set),
                "n_subword_tokens": len(s.token_ids),
                "n_eojeols": s.n_eojeols,
                "has_sentence_final": has_sentence_final_ending(s),
            }
            for s in selected
        ],
        "statistics": {
            "total_unique_morphemes_covered": len(all_morphemes),
            "avg_diversity_score": round(np.mean([s.diversity_score for s in selected]), 4),
            "morpheme_tag_distribution": tag_dist,
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[저장] Calibration set → {output_path}")
