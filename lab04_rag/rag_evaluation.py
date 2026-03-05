# rag_evaluation.py
"""
RAG 평가 — Azure AI Search 인덱스 활용 버전

변경 포인트:
  - 기존: from simple_rag import get_rag  (로컬 코사인 유사도)
  - 변경: from azure_rag import get_rag   (Azure AI Search 하이브리드 검색)

검색 모드를 변경하려면 아래 SEARCH_MODE를 수정하세요.
"""

import os
import sys
import json
import time
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab04_rag
sys.path.insert(0, project_root)

from lab02_baseline.baseline_test_data import get_test_cases
from lab02_baseline.baseline_evaluation import evaluate_response
from lab03_prompt_engineering.prompts import SYSTEM_PROMPT_V2

# ★ 핵심 변경: simple_rag → azure_rag ★
from azure_rag import get_rag

# Load .env from shared directory
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab08_evaluation
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)

# Use Azure Entra ID authentication instead of API key
credential = DefaultAzureCredential()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ── 검색 모드 선택 ──
# "vector"   : 벡터 검색만 (시맨틱 유사도)
# "keyword"  : 키워드 검색만 (BM25)
# "hybrid"   : 키워드 + 벡터 결합 (권장, 가장 정확)
# "semantic" : 시맨틱 랭킹 (AI Search 시맨틱 랭커, S2 이상 필요)
SEARCH_MODE = "hybrid"


def run_rag_test(test_case, deployment_name, rag_engine):
    """Azure AI Search RAG가 적용된 테스트"""

    # 1. 검색 모드에 따라 관련 문서 검색
    if SEARCH_MODE == "vector":
        search_results = rag_engine.search_vector(test_case["input"], top_k=2)
    elif SEARCH_MODE == "hybrid":
        search_results = rag_engine.search_hybrid(test_case["input"], top_k=2)
    elif SEARCH_MODE == "semantic":
        search_results = rag_engine.search_semantic(test_case["input"], top_k=2)
    else:  # keyword
        search_results = rag_engine.search(test_case["input"], top_k=2)

    # 2. 검색된 문서를 컨텍스트로 구성
    context = "\n\n---\n\n".join([
        f"[참고 문서: {r['title']}] (관련도: {r['score']:.4f})\n{r['content']}"
        for r in search_results
    ])

    # 3. 프롬프트 구성 (시스템 프롬프트 + RAG 컨텍스트)
    user_message = f"""다음 고객의 테니스 연습 관찰 결과를 분석하고 코칭 피드백을 제공해주세요.

## 참고 자료 (Azure AI Search에서 검색된 코칭 가이드)
{context}

## 고객 관찰 결과
{test_case['input']}

위 참고 자료를 바탕으로 전문적이고 구체적인 코칭 피드백을 작성해주세요."""

    start_time = time.time()

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1500,
        temperature=0.7,
    )

    elapsed = time.time() - start_time
    result = response.choices[0].message.content

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "response": result,
        "latency_seconds": round(elapsed, 2),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "search_mode": SEARCH_MODE,
        "rag_docs_used": [r["title"] for r in search_results],
        "rag_scores": [r["score"] for r in search_results],
    }


def run_rag_evaluation():
    """RAG 전체 평가 실행"""
    test_cases = get_test_cases()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    rag_engine = get_rag()
    results = []

    print(f"\n{'='*60}")
    print(f"🎾 RAG 평가 — Azure AI Search ({SEARCH_MODE} 모드)")
    print(f"{'='*60}\n")

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']})...", end=" ")

        result = run_rag_test(tc, deployment, rag_engine)
        evaluation = evaluate_response(result, tc)
        result["evaluation"] = evaluation
        results.append(result)

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_seconds"]

        # 검색된 문서 제목도 출력
        docs = ", ".join(result["rag_docs_used"])
        print(f"점수: {evaluation['score']}/100, 지연: {result['latency_seconds']}s")
        print(f"        검색 문서: {docs}")

    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)

    # 비용: OpenAI 추론 + 임베딩 + AI Search 호스팅
    input_cost = (total_input_tokens / 1_000_000) * 0.10
    output_cost = (total_output_tokens / 1_000_000) * 0.40
    embedding_cost = 0.002  # 검색 쿼리 임베딩 비용 (약 10회)
    total_cost = input_cost + output_cost + embedding_cost

    summary = {
        "label": f"rag_aisearch_{SEARCH_MODE}",
        "search_mode": SEARCH_MODE,
        "avg_score": round(avg_score, 1),
        "min_score": min(r["evaluation"]["score"] for r in results),
        "max_score": max(r["evaluation"]["score"] for r in results),
        "avg_latency": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "embedding_cost_usd": embedding_cost,
        "note": f"Azure AI Search {SEARCH_MODE} 검색 사용 (AI Search 호스팅 비용 별도)",
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"📊 결과 [{SEARCH_MODE.upper()} 모드]")
    print(f"   평균 점수: {avg_score:.1f}/100")
    print(f"   추론 비용: ${total_cost:.4f}")
    print(f"   평균 지연: {avg_latency:.2f}s")
    print(f"{'='*60}\n")

    output_file = f"eval_rag_{SEARCH_MODE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"💾 결과 저장: {output_file}")
    return summary


if __name__ == "__main__":
    run_rag_evaluation()