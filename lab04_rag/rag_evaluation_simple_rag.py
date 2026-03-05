# rag_evaluation_simple_rag.py
"""
RAG 평가 — 로컬 코사인 유사도 버전

변경 포인트:
  - 기존: from azure_rag import get_rag  (Azure AI Search 하이브리드 검색)
  - 변경: from simple_rag import get_rag  (로컬 코사인 유사도)

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

# ★ 핵심 변경: azure_rag → simple_rag ★
from simple_rag import get_rag

# Load .env from shared directory
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)

# Use Azure Entra ID authentication instead of API key
credential = DefaultAzureCredential()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def run_rag_test(test_case, deployment_name, rag_engine):
    """RAG가 적용된 테스트"""
    # 1. 관련 문서 검색
    search_results = rag_engine.search(test_case["input"], top_k=2)
    
    # 2. 컨텍스트 구성
    context = "\n\n---\n\n".join([
        f"[참고 문서: {r['title']}]\n{r['content']}" 
        for r in search_results
    ])
    
    # 3. 프롬프트 구성 (시스템 프롬프트 + RAG 컨텍스트)
    user_message = f"""다음 고객의 테니스 연습 관찰 결과를 분석하고 코칭 피드백을 제공해주세요.

## 참고 자료 (코칭 가이드에서 발췌)
{context}

## 고객 관찰 결과
{test_case['input']}

위 참고 자료를 바탕으로 전문적이고 구체적인 코칭 피드백을 작성해주세요."""

    start_time = time.time()
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1500,
        temperature=0.7
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
        "rag_docs_used": [r["title"] for r in search_results],
        "rag_scores": [r["score"] for r in search_results]
    }

def run_rag_evaluation():
    """RAG 전체 평가"""
    test_cases = get_test_cases()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    rag_engine = get_rag()
    results = []
    
    print(f"\n{'='*60}")
    print(f"🎾 RAG 적용 평가")
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
        
        print(f"점수: {evaluation['score']}/100, 지연: {result['latency_seconds']}s")
    
    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    # RAG는 임베딩 비용도 추가
    input_cost = (total_input_tokens / 1_000_000) * 0.10
    output_cost = (total_output_tokens / 1_000_000) * 0.40
    embedding_cost = 0.002  # 대략적 임베딩 비용
    total_cost = input_cost + output_cost + embedding_cost
    
    summary = {
        "label": "rag",
        "avg_score": round(avg_score, 1),
        "min_score": min(r["evaluation"]["score"] for r in results),
        "max_score": max(r["evaluation"]["score"] for r in results),
        "avg_latency": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "embedding_cost_usd": embedding_cost,
        "results": results
    }
    
    print(f"\n📊 결과: 평균 {avg_score:.1f}점 | 비용 ${total_cost:.4f} | 지연 {avg_latency:.2f}s\n")
    
    with open("eval_rag.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

if __name__ == "__main__":
    run_rag_evaluation()