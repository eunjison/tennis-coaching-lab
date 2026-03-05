# prompt_engineering_test.py
import os
import sys
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab03_prompt_engineering
sys.path.insert(0, project_root)

from lab02_baseline.baseline_test_data import get_test_cases
from lab02_baseline.baseline_evaluation import evaluate_response, run_full_evaluation
from prompts import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2
import json
import time

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

def run_prompt_engineering_test(test_case, deployment_name, system_prompt, label):
    """프롬프트 엔지니어링이 적용된 테스트"""
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"다음 고객의 테니스 연습 관찰 결과를 분석하고 코칭 피드백을 제공해주세요:\n\n{test_case['input']}"}
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
        "total_tokens": response.usage.total_tokens
    }

def run_pe_evaluation(system_prompt, label):
    """프롬프트 엔지니어링 전체 평가"""
    test_cases = get_test_cases()
    results = []
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    print(f"\n{'='*60}")
    print(f"🎾 프롬프트 엔지니어링 평가 - [{label.upper()}]")
    print(f"{'='*60}\n")
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0
    
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']})...", end=" ")
        
        result = run_prompt_engineering_test(tc, deployment, system_prompt, label)
        evaluation = evaluate_response(result, tc)
        result["evaluation"] = evaluation
        results.append(result)
        
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_seconds"]
        
        print(f"점수: {evaluation['score']}/100, 지연: {result['latency_seconds']}s")
    
    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    # 비용: 시스템 프롬프트 토큰이 추가됨
    input_cost = (total_input_tokens / 1_000_000) * 0.10
    output_cost = (total_output_tokens / 1_000_000) * 0.40
    total_cost = input_cost + output_cost
    
    summary = {
        "label": label,
        "avg_score": round(avg_score, 1),
        "min_score": min(r["evaluation"]["score"] for r in results),
        "max_score": max(r["evaluation"]["score"] for r in results),
        "avg_latency": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "system_prompt_length": len(system_prompt),
        "results": results
    }
    
    print(f"\n📊 결과: 평균 {avg_score:.1f}점 | 비용 ${total_cost:.4f} | 지연 {avg_latency:.2f}s\n")
    
    with open(f"eval_{label}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

if __name__ == "__main__":
    # V1: 기본 시스템 프롬프트
    summary_v1 = run_pe_evaluation(SYSTEM_PROMPT_V1, "prompt_v1")
    
    # V2: Few-shot 예제 포함
    summary_v2 = run_pe_evaluation(SYSTEM_PROMPT_V2, "prompt_v2")