# compare_results.py
import json
import os

def load_eval(filename):
    """평가 결과 로드"""
    if not os.path.exists(filename):
        print(f"⚠️ {filename} 파일이 없습니다. 해당 평가를 먼저 실행하세요.")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_all():
    """모든 단계 비교"""
    evaluations = {
        "Baseline (프롬프트 없음)": load_eval("eval_baseline.json"),
        "프롬프트 엔지니어링 V1": load_eval("eval_prompt_v1.json"),
        "프롬프트 엔지니어링 V2 (Few-shot)": load_eval("eval_prompt_v2.json"),
        "RAG + 프롬프트 V2": load_eval("eval_rag.json"),
    }
    
    # None 제거
    evaluations = {k: v for k, v in evaluations.items() if v is not None}
    
    if not evaluations:
        print("❌ 평가 결과 파일이 없습니다.")
        return
    
    print("\n" + "=" * 90)
    print("📊 테니스 코칭 AI 성능 비교 대시보드")
    print("=" * 90)
    
    # 헤더
    print(f"\n{'단계':<30} {'평균점수':>8} {'최저':>6} {'최고':>6} {'지연(s)':>8} {'토큰':>10} {'비용($)':>8}")
    print("-" * 90)
    
    baseline_score = None
    for label, data in evaluations.items():
        if baseline_score is None:
            baseline_score = data["avg_score"]
            delta = ""
        else:
            diff = data["avg_score"] - baseline_score
            delta = f" ({'+' if diff >= 0 else ''}{diff:.1f})"
        
        print(f"{label:<30} {data['avg_score']:>6.1f}{delta:>8} {data['min_score']:>6} "
              f"{data['max_score']:>6} {data['avg_latency']:>8.2f} "
              f"{data['total_tokens']:>10,} {data['estimated_cost_usd']:>8.4f}")
    
    # 카테고리별 분석
    print(f"\n\n{'='*90}")
    print("📈 카테고리별 점수 비교")
    print("=" * 90)
    
    categories = set()
    for data in evaluations.values():
        for r in data.get("results", []):
            categories.add(r["category"])
    
    print(f"\n{'카테고리':<25}", end="")
    for label in evaluations:
        short_label = label[:15]
        print(f" {short_label:>15}", end="")
    print()
    print("-" * 90)
    
    for cat in sorted(categories):
        print(f"{cat:<25}", end="")
        for label, data in evaluations.items():
            cat_scores = [r["evaluation"]["score"] for r in data.get("results", []) if r["category"] == cat]
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            print(f" {avg:>15.1f}", end="")
        print()
    
    # 비용 효율성 분석
    print(f"\n\n{'='*90}")
    print("💰 비용 효율성 분석")
    print("=" * 90)
    
    for label, data in evaluations.items():
        score_per_dollar = data["avg_score"] / max(data["estimated_cost_usd"], 0.0001)
        tokens_per_score = data["total_tokens"] / max(data["avg_score"], 1)
        print(f"\n{label}:")
        print(f"  점수당 비용: ${data['estimated_cost_usd'] / max(data['avg_score'], 1):.6f}/점")
        print(f"  달러당 점수: {score_per_dollar:.1f}점/$")
        print(f"  점수당 토큰: {tokens_per_score:.0f} tokens/점")
    
    # 종합 권장 사항
    print(f"\n\n{'='*90}")
    print("💡 종합 권장 사항")
    print("=" * 90)
    
    best = max(evaluations.items(), key=lambda x: x[1]["avg_score"])
    cheapest = min(evaluations.items(), key=lambda x: x[1]["estimated_cost_usd"])
    
    print(f"\n최고 점수: {best[0]} ({best[1]['avg_score']}점)")
    print(f"최저 비용: {cheapest[0]} (${cheapest[1]['estimated_cost_usd']:.4f})")
    
    if best[1]["avg_score"] < 80:
        print(f"\n⚠️ 현재 최고 점수가 {best[1]['avg_score']}점입니다.")
        print("Fine-tuning을 통해 추가 성능 향상이 필요합니다.")
        print("특히 다음 영역에서 Fine-tuning이 도움이 될 수 있습니다:")
        
        # 약한 카테고리 식별
        for cat in sorted(categories):
            for label, data in [best]:
                scores = [r["evaluation"]["score"] for r in data.get("results", []) if r["category"] == cat]
                if scores and sum(scores)/len(scores) < 70:
                    print(f"  - {cat}: 평균 {sum(scores)/len(scores):.0f}점")

if __name__ == "__main__":
    compare_all()