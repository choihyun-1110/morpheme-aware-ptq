"""
시각화 모듈
- 형태소-토큰 정렬(alignment) 시각화
- 다양성 점수 분포 시각화
"""
import os
from typing import List, Optional


def visualize_alignment(text: str, tokenizer, save_path: Optional[str] = None):
    """
    단일 문장에 대해 형태소 분석 결과와 LLM tokenizer 결과를 나란히 시각화.
    
    Args:
        text: 분석할 문장
        tokenizer: HuggingFace AutoTokenizer 인스턴스
        save_path: 저장 경로 (None이면 stdout에 출력)
    """
    from kiwipiepy import Kiwi
    
    kiwi = Kiwi()
    
    # 형태소 분석
    morph_result = kiwi.tokenize(text)
    morph_tokens = [(t.form, t.tag) for t in morph_result]
    
    # LLM 토큰화
    encoding = tokenizer(text, add_special_tokens=False)
    token_ids = encoding['input_ids']
    subword_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # 출력
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"원문: {text}")
    lines.append(f"{'='*70}")
    lines.append(f"\n[Kiwi 형태소] ({len(morph_tokens)}개)")
    morph_str = " | ".join([f"{form}/{tag}" for form, tag in morph_tokens])
    lines.append(f"  {morph_str}")
    
    lines.append(f"\n[LLM Tokenizer] ({len(subword_tokens)}개)")
    token_str = " | ".join(subword_tokens)
    lines.append(f"  {token_str}")
    
    # 어절별 비교
    lines.append(f"\n[어절별 분절 비교]")
    eojeols = text.split()
    for eojeol in eojeols:
        # 해당 어절의 subword 수
        eojeol_tokens = tokenizer(eojeol, add_special_tokens=False)['input_ids']
        decoded = [tokenizer.decode([tid]) for tid in eojeol_tokens]
        lines.append(f"  '{eojeol}' → {len(decoded)} subwords: {decoded}")
    
    lines.append(f"\n  UMR = {len(set(morph_tokens))} / {len(morph_tokens)} = "
                 f"{len(set(morph_tokens))/len(morph_tokens):.3f}")
    lines.append(f"  TTR = {len(set(token_ids))} / {len(token_ids)} = "
                 f"{len(set(token_ids))/len(token_ids):.3f}")
    lines.append(f"  SFS = {len(token_ids)} / {len(eojeols)} = "
                 f"{len(token_ids)/len(eojeols):.2f}")
    
    output = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"[시각화] → {save_path}")
    else:
        print(output)


def visualize_score_distribution(scores, save_path: Optional[str] = None):
    """
    다양성 점수 분포를 히스토그램으로 시각화.
    
    Args:
        scores: SentenceScore 리스트
        save_path: 이미지 저장 경로 (None이면 plt.show())
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 서버 환경용
    except ImportError:
        print("[시각화] matplotlib 미설치. 텍스트 통계만 출력합니다.")
        _print_stats_text(scores)
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Diversity Score Distribution', fontsize=16, fontweight='bold')
    
    # UMR 분포
    umrs = [s.umr for s in scores]
    axes[0, 0].hist(umrs, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    axes[0, 0].set_title('UMR (Unique Morpheme Ratio)')
    axes[0, 0].set_xlabel('UMR')
    axes[0, 0].axvline(sum(umrs)/len(umrs), color='red', linestyle='--', label=f'mean={sum(umrs)/len(umrs):.3f}')
    axes[0, 0].legend()
    
    # TTR 분포
    ttrs = [s.ttr for s in scores]
    axes[0, 1].hist(ttrs, bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    axes[0, 1].set_title('TTR (Type-Token Ratio)')
    axes[0, 1].set_xlabel('TTR')
    axes[0, 1].axvline(sum(ttrs)/len(ttrs), color='red', linestyle='--', label=f'mean={sum(ttrs)/len(ttrs):.3f}')
    axes[0, 1].legend()
    
    # SFS 분포
    sfss = [s.sfs for s in scores]
    axes[1, 0].hist(sfss, bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    axes[1, 0].set_title('SFS (Subword Fragmentation Score)')
    axes[1, 0].set_xlabel('SFS')
    axes[1, 0].axvline(sum(sfss)/len(sfss), color='red', linestyle='--', label=f'mean={sum(sfss)/len(sfss):.2f}')
    axes[1, 0].legend()
    
    # Composite Score 분포
    ds = [s.diversity_score for s in scores]
    axes[1, 1].hist(ds, bins=50, color='#9b59b6', alpha=0.7, edgecolor='white')
    axes[1, 1].set_title('Composite Diversity Score D(s)')
    axes[1, 1].set_xlabel('D(s)')
    axes[1, 1].axvline(sum(ds)/len(ds), color='red', linestyle='--', label=f'mean={sum(ds)/len(ds):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[시각화] 분포 그래프 → {save_path}")
    else:
        plt.show()
    
    plt.close()


def _print_stats_text(scores):
    """matplotlib 없을 때 텍스트 통계 출력."""
    import numpy as np
    print(f"\n--- 다양성 점수 통계 ---")
    print(f"  UMR: mean={np.mean([s.umr for s in scores]):.3f}, "
          f"std={np.std([s.umr for s in scores]):.3f}")
    print(f"  TTR: mean={np.mean([s.ttr for s in scores]):.3f}, "
          f"std={np.std([s.ttr for s in scores]):.3f}")
    print(f"  SFS: mean={np.mean([s.sfs for s in scores]):.2f}, "
          f"std={np.std([s.sfs for s in scores]):.2f}")
    print(f"  D:   mean={np.mean([s.diversity_score for s in scores]):.3f}, "
          f"std={np.std([s.diversity_score for s in scores]):.3f}")


if __name__ == "__main__":
    # 테스트용: 토크나이저 로드 없이 alignment 시각화 데모
    print("시각화 모듈 로드 완료. 사용법:")
    print("  from visualize import visualize_alignment")
    print("  visualize_alignment('한국어 문장입니다.', tokenizer)")
