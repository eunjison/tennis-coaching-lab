# limitation_analysis.py
import json
import os

def analyze_limitations():
    """프롬프트 엔지니어링 + RAG의 한계점 분석"""
    
    # 최선의 결과 (RAG 또는 Prompt V2) 로드
    best_file = "eval_rag.json" if os.path.exists("eval_rag.json") else "eval_prompt_v2.json"
    with open(best_file, "r", encoding="utf-8") as f:
        best_data = json.load(f)
    
    print("\n" + "=" * 70)
    print("🔍 프롬프트 엔지니어링 + RAG 한계점 분석")
    print("=" * 70)
    
    # 1. 낮은 점수 케이스 분석
    print("\n📌 1. 성능이 부족한 테스트 케이스")
    print("-" * 50)
    
    weak_cases = [r for r in best_data["results"] if r["evaluation"]["score"] < 75]
    for case in weak_cases:
        print(f"\n  [{case['id']}] {case['category']} - 점수: {case['evaluation']['score']}")
        for criterion, value in case["evaluation"]["details"].items():
            if "❌" in str(value):
                print(f"    ❌ 미충족: {criterion}")
    
    # 2. 구조적 한계
    print(f"\n\n📌 2. 프롬프트 엔지니어링의 구조적 한계")
    print("-" * 50)
    
    limitations_pe = [
        {
            "issue": "응답 형식 일관성 부족",
            "description": "시스템 프롬프트에 형식을 지정해도 모델이 때때로 다른 형식으로 응답합니다.",
            "example": "잘하고 있는 점을 생략하거나, 레퍼런스를 빠뜨리는 경우가 있습니다.",
            "ft_solution": "Fine-tuning 데이터에서 일관된 형식을 학습시키면 해결됩니다."
        },
        {
            "issue": "도메인 전문성 깊이 부족",
            "description": "테니스 전문 용어와 기술 설명의 정확도가 일반적인 수준에 머뭅니다.",
            "example": "'세미웨스턴 그립의 V자 위치'처럼 구체적인 기술 묘사가 부족합니다.",
            "ft_solution": "전문가 수준의 코칭 데이터로 Fine-tuning하면 전문성이 향상됩니다."
        },
        {
            "issue": "코칭 톤/스타일 불안정",
            "description": "때로는 너무 학술적이거나, 때로는 너무 간략한 피드백을 생성합니다.",
            "example": "초보자에게 과도하게 기술적인 용어를 사용하거나, 상급자에게 기본적인 조언만 하는 경우",
            "ft_solution": "레벨별 맞춤 피드백 예시로 Fine-tuning하면 일관된 톤을 유지합니다."
        },
        {
            "issue": "시스템 프롬프트 토큰 오버헤드",
            "description": "긴 시스템 프롬프트로 인해 매 요청마다 추가 토큰 비용이 발생합니다.",
            "example": f"현재 시스템 프롬프트만으로 약 500~1000 토큰이 소비됩니다.",
            "ft_solution": "Fine-tuning 후 짧은 시스템 프롬프트(또는 없이)로도 동일 품질 달성 가능합니다."
        }
    ]
    
    for i, lim in enumerate(limitations_pe, 1):
        print(f"\n  {i}. {lim['issue']}")
        print(f"     설명: {lim['description']}")
        print(f"     예시: {lim['example']}")
        print(f"     FT 해결: {lim['ft_solution']}")
    
    # 3. RAG 한계
    print(f"\n\n📌 3. RAG의 구조적 한계")
    print("-" * 50)
    
    limitations_rag = [
        {
            "issue": "검색 정확도 의존",
            "description": "쿼리와 문서의 의미적 매칭이 항상 정확하지 않습니다.",
            "impact": "관련 없는 문서가 검색되면 오히려 품질이 떨어질 수 있습니다."
        },
        {
            "issue": "컨텍스트 길이 증가",
            "description": "RAG 문서가 추가되면 입력 토큰이 크게 증가합니다.",
            "impact": "비용 증가 + 지연시간 증가. 실시간 서비스에서 부담이 됩니다."
        },
        {
            "issue": "지식 통합의 한계",
            "description": "모델이 RAG 문서를 단순 인용할 뿐, 진정한 전문 지식으로 내재화하지 못합니다.",
            "impact": "문서에 없는 상황에 대한 응용력이 부족합니다."
        },
        {
            "issue": "인프라 복잡성",
            "description": "AI Search 서비스 운영, 임베딩 업데이트, 문서 관리 등 추가 인프라가 필요합니다.",
            "impact": "운영 비용과 복잡성이 증가합니다."
        }
    ]
    
    for i, lim in enumerate(limitations_rag, 1):
        print(f"\n  {i}. {lim['issue']}")
        print(f"     설명: {lim['description']}")
        print(f"     영향: {lim['impact']}")
    
    # 4. Fine-tuning으로 해결 가능한 영역 요약
    print(f"\n\n📌 4. Fine-tuning으로 기대되는 개선 사항")
    print("-" * 50)
    print("""
  ✅ 응답 형식의 100% 일관성 (학습 데이터의 형식을 따름)
  ✅ 도메인 전문 지식의 내재화 (RAG 없이도 전문적 피드백)
  ✅ 레벨별 맞춤 코칭 톤 (학습 데이터에서 패턴 습득)
  ✅ 시스템 프롬프트 토큰 절약 (짧은 프롬프트로 동일 품질)
  ✅ 지연시간 감소 (RAG 검색 단계 생략 가능)
  ✅ 인프라 단순화 (AI Search 불필요)
  
  ⚠️ Fine-tuning의 트레이드오프:
  - 학습 비용 발생 (1회성, 약 $5~15)
  - 학습 데이터 준비에 시간 소요
  - 새로운 지식 추가 시 재학습 필요
  - 모델 업데이트 시 재학습 필요
    """)
    
    return {
        "weak_cases": len(weak_cases),
        "pe_limitations": len(limitations_pe),
        "rag_limitations": len(limitations_rag)
    }

if __name__ == "__main__":
    analyze_limitations()