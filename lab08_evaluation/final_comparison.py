# final_comparison.py
import json
import os

def final_comparison():
    """최종 전체 비교"""
    
    files = {
        "① Baseline": "../lab02_baseline/eval_baseline.json",
        "② 프롬프트 V1": "../lab03_prompt_engineering/eval_prompt_v1.json", 
        "③ 프롬프트 V2 (Few-shot)": "../lab03_prompt_engineering/eval_prompt_v2.json",
        "④ RAG + 프롬프트 V2": "../lab04_rag/eval_rag.json",
        "⑤ Fine-tuned": "eval_finetuned.json"
    }
    
    data = {}
    for label, f in files.items():
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fh:
                data[label] = json.load(fh)
    
    print("\n" + "=" * 100)
    print("🏆 최종 성능 비교 대시보드 - 테니스 코칭 AI")
    print("=" * 100)
    
    # 주요 지표 비교
    print(f"\n{'단계':<30} {'평균점수':>8} {'변화':>8} {'지연(s)':>8} {'토큰':>10} {'추론비용':>10}")
    print("-" * 100)
    
    baseline_score = None
    for label, d in data.items():
        if baseline_score is None:
            baseline_score = d["avg_score"]
            delta = "-"
        else:
            diff = d["avg_score"] - baseline_score
            delta = f"{'+' if diff >= 0 else ''}{diff:.1f}"
        
        print(f"{label:<30} {d['avg_score']:>6.1f}점 {delta:>8} {d['avg_latency']:>7.2f}s "
              f"{d['total_tokens']:>10,} ${d.get('estimated_cost_usd', d.get('estimated_inference_cost_usd', 0)):>9.4f}")
    
    # Fine-tuning 투자 분석
    if "⑤ Fine-tuned" in data and len(data) > 1:
        ft = data["⑤ Fine-tuned"]
        non_ft_evaluations = [(k, v) for k, v in data.items() if k != "⑤ Fine-tuned"]
        
        if non_ft_evaluations:  # Only proceed if there are non-fine-tuned evaluations
            best_non_ft = max(non_ft_evaluations, key=lambda x: x[1]["avg_score"])
            
            print(f"\n\n{'='*100}")
            print("💰 Fine-tuning 투자 분석 (ROI)")
            print("=" * 100)
            
            score_improvement = ft["avg_score"] - best_non_ft[1]["avg_score"]
            training_cost = ft.get("training_cost_usd", 10)
            
            # 추론 비용 비교 (per request)
            ft_per_request = ft.get("estimated_inference_cost_usd", ft.get("estimated_cost_usd", 0)) / 10
            best_per_request = best_non_ft[1].get("estimated_cost_usd", 0) / 10
            
            # 토큰 절약 (시스템 프롬프트 축소)
            ft_tokens = ft["total_tokens"]
            best_tokens = best_non_ft[1]["total_tokens"]
            token_savings = best_tokens - ft_tokens
            
            print(f"\n비교 대상: {best_non_ft[0]} → ⑤ Fine-tuned")
            print(f"  점수 변화: {best_non_ft[1]['avg_score']:.1f} → {ft['avg_score']:.1f} ({'+' if score_improvement >= 0 else ''}{score_improvement:.1f}점)")
            print(f"  토큰 변화: {best_tokens:,} → {ft_tokens:,} ({'+' if token_savings < 0 else '-'}{abs(token_savings):,} tokens)")
            print(f"  지연 변화: {best_non_ft[1]['avg_latency']:.2f}s → {ft['avg_latency']:.2f}s")
            print(f"  추론비용/건: ${best_per_request:.4f} → ${ft_per_request:.4f}")
            
            print(f"\n  📌 Fine-tuning 학습 비용: ${training_cost:.2f} (1회성)")
            
            if ft_per_request < best_per_request:
                cost_saving_per_request = best_per_request - ft_per_request
                breakeven = int(training_cost / cost_saving_per_request)
                print(f"  📌 요청당 절약: ${cost_saving_per_request:.4f}")
                print(f"  📌 손익분기점: {breakeven:,}건 요청 후 학습 비용 회수")
            else:
                extra_cost = ft_per_request - best_per_request
                print(f"  ⚠️ 추론 비용 증가: 요청당 +${extra_cost:.4f}")
                print(f"  ⚠️ 그러나 품질 향상({score_improvement:+.1f}점)으로 고객 만족도 증가 기대")
    
    # 종합 권장 사항
    print(f"\n\n{'='*100}")
    print("📋 최종 권장 사항")
    print("=" * 100)
    
    print("""
    🏅 프로덕션 환경 권장 구성:
    
    ┌─ 일반 고객 피드백 ───── Fine-tuned GPT-4.1-nano (비용 효율)
    │
    ├─ 상세 기술 분석 ──── Fine-tuned + RAG (전문 지식 보강 필요 시)
    │
    └─ 실시간 영상 분석 ── GPT-4.1-nano Vision (프레임 분석)
                          + Fine-tuned GPT-4.1-nano (코칭 생성)
    
    💡 팁:
    - 새로운 기술/트렌드 반영 시: RAG 문서 업데이트 (즉시 반영)
    - 코칭 스타일 변경 시: Fine-tuning 재학습 (1~2시간)
    - 비용 절감: 배치 처리 시 Batch API 활용 (50% 할인)
    """)

if __name__ == "__main__":
    final_comparison()