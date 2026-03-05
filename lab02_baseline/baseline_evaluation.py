# baseline_evaluation.py
import os
import sys
import json
import time
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Handle import for baseline_test_data (works both when run directly and when imported)
try:
    from baseline_test_data import get_test_cases
except ImportError:
    # If imported from outside, try absolute import
    from lab02_baseline.baseline_test_data import get_test_cases

# Load .env from shared directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab08_evaluation
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)

# Use Azure Entra ID authentication instead of API key
credential = DefaultAzureCredential()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def run_baseline_test(test_case, deployment_name):
    """기본 모델로 테니스 코칭 피드백 생성"""
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user", "content": f"다음 테니스 연습 관찰 결과를 분석하고 피드백을 주세요:\n\n{test_case['input']}"}
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

def evaluate_response(result, test_case):
    """응답 품질을 자동 평가 (0~100점)"""
    criteria = test_case["evaluation_criteria"]
    score = 0
    max_score = 0
    details = {}
    
    response_lower = result["response"].lower()
    response_text = result["response"]
    
    # 1. 긍정적 피드백 포함 여부 (15점)
    max_score += 15
    positive_keywords = ["잘하", "좋은", "훌륭", "좋습니다", "잘 하", "장점", "강점"]
    has_positive = any(kw in response_text for kw in positive_keywords)
    if has_positive and criteria.get("identifies_positive"):
        score += 15
        details["positive_feedback"] = "✅ 포함"
    else:
        details["positive_feedback"] = "❌ 미포함"
    
    # 2. 주요 이슈 식별 (30점)
    issues = criteria.get("identifies_issues", [])
    if issues:
        max_score += 30
        found = sum(1 for issue in issues if issue in response_text)
        issue_score = int((found / len(issues)) * 30)
        score += issue_score
        details["issues_identified"] = f"{found}/{len(issues)} ({issue_score}점)"
    
    # 3. 연습 방법 제공 (15점)
    max_score += 15
    drill_keywords = ["연습", "드릴", "훈련", "반복", "세트", "practice"]
    has_drill = any(kw in response_text for kw in drill_keywords)
    if has_drill and criteria.get("provides_drills"):
        score += 15
        details["drills_provided"] = "✅ 포함"
    else:
        details["drills_provided"] = "❌ 미포함"
    
    # 4. 레퍼런스/참고자료 (10점)
    max_score += 10
    ref_keywords = ["참고", "레퍼런스", "영상", "선수", "참조", "추천"]
    has_ref = any(kw in response_text for kw in ref_keywords)
    if has_ref and criteria.get("provides_reference"):
        score += 10
        details["reference_provided"] = "✅ 포함"
    else:
        details["reference_provided"] = "❌ 미포함"
    
    # 5. 구체적 수치 사용 (15점)
    max_score += 15
    import re
    has_numbers = bool(re.search(r'\d+[cm도%km회초분m]', response_text))
    if has_numbers:
        score += 15
        details["specific_numbers"] = "✅ 포함"
    else:
        details["specific_numbers"] = "❌ 미포함"
    
    # 6. 격려적 톤 (15점)
    max_score += 15
    encouraging = ["화이팅", "발전", "향상", "개선", "좋아질", "가능", "할 수 있", "시작"]
    has_encourage = any(kw in response_text for kw in encouraging)
    if has_encourage:
        score += 15
        details["encouraging_tone"] = "✅ 포함"
    else:
        details["encouraging_tone"] = "❌ 미포함"
    
    # 정규화
    normalized_score = int((score / max_score) * 100) if max_score > 0 else 0
    
    return {
        "score": normalized_score,
        "details": details,
        "raw_score": score,
        "max_score": max_score
    }

def run_full_evaluation(deployment_name, label="baseline"):
    """전체 테스트 케이스 평가 실행"""
    test_cases = get_test_cases()
    results = []
    
    print(f"\n{'='*60}")
    print(f"🎾 테니스 코칭 AI 평가 - [{label.upper()}]")
    print(f"모델: {deployment_name}")
    print(f"테스트 케이스: {len(test_cases)}개")
    print(f"{'='*60}\n")
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0
    
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']})...", end=" ")
        
        result = run_baseline_test(tc, deployment_name)
        evaluation = evaluate_response(result, tc)
        
        result["evaluation"] = evaluation
        results.append(result)
        
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_seconds"]
        
        print(f"점수: {evaluation['score']}/100, 지연: {result['latency_seconds']}s")
    
    # 종합 리포트
    avg_score = sum(r["evaluation"]["score"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    
    # 비용 계산 (GPT-4.1-nano Global Standard 기준)
    input_cost = (total_input_tokens / 1_000_000) * 0.10
    output_cost = (total_output_tokens / 1_000_000) * 0.40
    total_cost = input_cost + output_cost
    
    summary = {
        "label": label,
        "model": deployment_name,
        "avg_score": round(avg_score, 1),
        "min_score": min(r["evaluation"]["score"] for r in results),
        "max_score": max(r["evaluation"]["score"] for r in results),
        "avg_latency": round(avg_latency, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "results": results
    }
    
    print(f"\n{'='*60}")
    print(f"📊 종합 결과 [{label.upper()}]")
    print(f"{'='*60}")
    print(f"평균 점수: {avg_score:.1f}/100")
    print(f"최저/최고: {summary['min_score']}/{summary['max_score']}")
    print(f"평균 지연시간: {avg_latency:.2f}s")
    print(f"총 토큰 사용: {summary['total_tokens']:,}")
    print(f"  - 입력: {total_input_tokens:,} tokens")
    print(f"  - 출력: {total_output_tokens:,} tokens")
    print(f"예상 비용: ${total_cost:.4f}")
    print(f"{'='*60}\n")
    
    # 결과 저장
    with open(f"eval_{label}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

if __name__ == "__main__":
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-41-nano-base")
    summary = run_full_evaluation(deployment, "baseline")