"""
A-2: 나무위키(B/C_v3) vs Wikitext-2(A/C_en_v3) 다양성 지표 비교
목적: C_en_v3 ≈ A 원인 설명 (Wikitext-2가 이미 균질)
"""
import json, glob, collections, re, sys

R = "/home/choihyun/workspace/results"


def load_sentences(path):
    with open(path) as f:
        d = json.load(f)
    sents = d.get("sentences", [])
    return [s["text"] if isinstance(s, dict) else s for s in sents]


def tokenize_en(texts):
    tokens = []
    for t in texts:
        tokens.extend(re.findall(r"[a-zA-Z']+", t.lower()))
    return tokens


def tokenize_ko(texts):
    tokens = []
    for t in texts:
        tokens.extend(t.split())
    return tokens


def diversity_stats(sents, lang="en"):
    tokens = tokenize_en(sents) if lang == "en" else tokenize_ko(sents)
    total = len(tokens)
    unique = len(set(tokens))
    ttr = unique / total if total > 0 else 0

    freq = collections.Counter(tokens)
    sorted_freq = sorted(freq.values(), reverse=True)
    top10_types = max(1, len(sorted_freq) // 10)
    top10_coverage = sum(sorted_freq[:top10_types]) / total if total > 0 else 0

    avg_len = sum(len(s.split()) for s in sents) / len(sents) if sents else 0

    return {
        "n_sentences": len(sents),
        "total_tokens": total,
        "unique_tokens": unique,
        "TTR": round(ttr, 4),
        "top10pct_coverage": round(top10_coverage, 4),
        "avg_words_per_sent": round(avg_len, 1),
    }


def load_all():
    a = load_sentences(R + "/calibration_set_A.json")
    b = load_sentences(R + "/calibration_set_B.json")
    cv3 = load_sentences(R + "/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json")
    cen_files = sorted(glob.glob(R + "/calibration_set_C_en_v3*.json"))
    cen = load_sentences(cen_files[0]) if cen_files else []
    return a, b, cv3, cen


def main():
    a, b, cv3, cen = load_all()

    a_s   = diversity_stats(a,   lang="en")
    cen_s = diversity_stats(cen, lang="en")
    b_s   = diversity_stats(b,   lang="ko")
    cv3_s = diversity_stats(cv3, lang="ko")

    print("\n=== 다양성 지표 비교 (선별된 128문장 기준) ===\n")
    header = "{:<28} {:>14} {:>14} {:>14} {:>14}".format(
        "지표", "A(wikitext)", "C_en_v3", "B(나무위키)", "C_v3"
    )
    print(header)
    print("-" * 75)
    metrics = [
        ("n_sentences",       "문장 수"),
        ("total_tokens",      "총 토큰 수"),
        ("unique_tokens",     "고유 토큰 수"),
        ("TTR",               "TTR (다양성)"),
        ("top10pct_coverage", "상위10% 집중도"),
        ("avg_words_per_sent","평균 문장 길이"),
    ]
    for key, label in metrics:
        row = "{:<28} {:>14} {:>14} {:>14} {:>14}".format(
            label,
            str(a_s[key]),
            str(cen_s[key]),
            str(b_s[key]),
            str(cv3_s[key]),
        )
        print(row)

    print()
    print("=== 핵심 비교 ===")
    a_ttr   = a_s["TTR"]
    cen_ttr = cen_s["TTR"]
    b_ttr   = b_s["TTR"]
    cv3_ttr = cv3_s["TTR"]

    print(f"[영어] A TTR={a_ttr:.4f}  →  C_en_v3 TTR={cen_ttr:.4f}  (변화: {cen_ttr-a_ttr:+.4f})")
    print(f"[한국어] B TTR={b_ttr:.4f}  →  C_v3 TTR={cv3_ttr:.4f}  (변화: {cv3_ttr-b_ttr:+.4f})")
    print()
    print(f"[영어] A 집중도={a_s['top10pct_coverage']:.4f}  →  C_en_v3 집중도={cen_s['top10pct_coverage']:.4f}")
    print(f"[한국어] B 집중도={b_s['top10pct_coverage']:.4f}  →  C_v3 집중도={cv3_s['top10pct_coverage']:.4f}")
    print()

    print("=== 해석 ===")
    en_improvement = cen_ttr - a_ttr
    ko_improvement = cv3_ttr - b_ttr
    print(f"greedy selection TTR 개선: 영어 {en_improvement:+.4f} vs 한국어 {ko_improvement:+.4f}")
    if abs(en_improvement) < abs(ko_improvement):
        print("→ Wikitext-2는 랜덤 선별(A)과 greedy 선별(C_en_v3)의 TTR 차이가 미미")
        print("→ 나무위키는 greedy 선별이 TTR을 크게 높임 → 다양성 알고리즘 효과가 큼")
        print("→ C_en_v3 ≈ A 원인: Wikitext-2 자체가 이미 균질한 고품질 텍스트 (천장 효과)")
    else:
        print("→ 예상과 다른 결과. 추가 분석 필요.")

    # Also check source pool diversity if available
    print()
    print("참고: 이 비교는 선별된 128문장 기준.")
    print("실제로는 원본 풀(wikitext 전체 vs 나무위키 전체)의 TTR 비교가 더 정확하나,")
    print("선별된 결과로도 greedy selection의 상대적 이득 방향은 확인 가능.")


if __name__ == "__main__":
    main()
