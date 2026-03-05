# finetuned_evaluation.py
import os
import sys
import json
import time
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab08_evaluation
sys.path.insert(0, project_root)

from lab02_baseline.baseline_test_data import get_test_cases
from lab02_baseline.baseline_evaluation import evaluate_response

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

def run_ft_test(test_case, deployment_name):
    """Fine-tuned 모델 테스트 - 짧은 시스템 프롬프트 사용"""
    start_time = time.time()
    
    # Fine-tuned 모델은 짧은 시스템 프롬프트만으로 충분
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "당신은 KTA 인증 테니스 코치입니다. 고객의 연습 관찰 결과를 분석하여 전문적이고 격려적인 코칭 피드백을 제공합니다."},
            {"role": "user", "content": f"고객 관찰: {test_case['input']}"}
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

def run_ft_evaluation():
    """Fine-tuned 모델 전체 평가"""
    test_cases = get_test_cases()
    ft_deployment = os.getenv("AZURE_OPENAI_FT_DEPLOYMENT", "gpt-41-nano-tennis-coach")
    results = []
    
    print(f"\n{'='*60}")
    print(f"🎾 Fine-tuned 모델 평가")
    print(f"모델: {ft_deployment}")
    print(f"{'='*60}\n")
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0
    
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']})...", end=" ")
        
        result = run_ft_test(tc, ft_deployment)
        evaluation = evaluate_response(result, tc)
        result["evaluation"] = evaluation
        results.append(result)
        
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_seconds"]
        
        print(f"점수: {evaluation['score']}/100, 지연: {result['latency_seconds']}s")
    
    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    # Fine-tuned 모델 비용 (추론 비용은 base 모델보다 약간 높을 수 있음)
    # GPT-4.1-nano FT: 약 $0.20 input / $0.80 output per 1M tokens (예상)
    input_cost = (total_input_tokens / 1_000_000) * 0.20
    output_cost = (total_output_tokens / 1_000_000) * 0.80
    total_cost = input_cost + output_cost
    
    # Fine-tuning 학습 비용 (1회성)
    training_cost_estimate = 10.0  # 약 $5~15 예상
    
    summary = {
        "label": "fine-tuned",
        "avg_score": round(avg_score, 1),
        "min_score": min(r["evaluation"]["score"] for r in results),
        "max_score": max(r["evaluation"]["score"] for r in results),
        "avg_latency": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_inference_cost_usd": round(total_cost, 4),
        "training_cost_usd": training_cost_estimate,
        "note": "Fine-tuned 모델은 짧은 시스템 프롬프트를 사용하여 입력 토큰이 적음",
        "results": results
    }
    
    print(f"\n📊 결과: 평균 {avg_score:.1f}점 | 추론비용 ${total_cost:.4f} | 지연 {avg_latency:.2f}s")
    print(f"📊 학습비용: ~${training_cost_estimate:.2f} (1회성)")
    
    with open("eval_finetuned.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

if __name__ == "__main__":
    run_ft_evaluation()