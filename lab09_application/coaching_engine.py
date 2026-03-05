# coaching_engine.py
"""테니스 코칭 AI 엔진"""
import os
import json
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
from frame_extractor import frame_to_base64, get_demo_frames

# Load .env from shared directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab09_application
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)

# Use Azure Entra ID authentication instead of API key
credential = DefaultAzureCredential()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Fine-tuned 배포명 (Fine-tuning 전에는 base 모델 사용)
FT_DEPLOYMENT = os.getenv("AZURE_OPENAI_FT_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
BASE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def analyze_frame_with_vision(frame_path):
    """Vision API로 프레임 분석 (Base 모델 사용)"""
    base64_image = frame_to_base64(frame_path)
    
    response = client.chat.completions.create(
        model=BASE_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "당신은 테니스 전문 분석가입니다. 이미지에서 테니스 플레이어의 자세, 폼, 위치를 관찰하고 구조화된 관찰 결과를 작성하세요."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 테니스 연습 프레임을 분석해주세요. 다음 항목을 관찰하세요:\n1. 어떤 샷(스트로크/서브/발리)을 하고 있는지\n2. 그립\n3. 스탠스/자세\n4. 라켓 위치 및 스윙 궤적\n5. 몸의 회전\n6. 발 위치\n간결한 관찰 결과만 작성해주세요."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content

def analyze_demo_frames(demo_frames):
    """데모용: 텍스트 설명을 기반으로 전체 관찰 결과 생성"""
    observations = "\n".join([
        f"[{f['timestamp']}] {f['description']}" for f in demo_frames
    ])
    return observations

def generate_coaching_report(observations, use_finetuned=True):
    """관찰 결과를 바탕으로 코칭 리포트 생성"""
    deployment = FT_DEPLOYMENT if use_finetuned else BASE_DEPLOYMENT
    
    system_prompt = "당신은 KTA 인증 테니스 코치입니다. 고객의 연습 관찰 결과를 분석하여 전문적이고 격려적인 코칭 피드백을 제공합니다."
    
    user_prompt = f"""아래는 고객이 무인 테니스 연습장에서 30분 동안 연습한 영상을 분석한 관찰 결과입니다.
이 관찰 결과를 종합하여 고객에게 제공할 종합 코칭 리포트를 작성해주세요.

각 영역(포핸드, 백핸드, 서브, 발리, 풋워크, 체력)별로 분석합니다. 영상에서 수행하고 있는 샷과 자세를 기반으로, 각 영역에서 고객의 강점과 개선할 점을 구체적으로 지적해주세요.
포함되지 않은 영역은 분석에서 제외해주세요.
잘한 점은 칭찬과 함께 강조하고, 개선할 점은 구체적인 조언과 함께 제시해주세요. 또한, 가장 시급하게 개선해야 할 Top 3 과제를 선정하여 우선순위를 명확히 해주세요.
만약 개선해야할 점을 발견하지 못했다면, 분석에서 제외해 주세요.

## 30분 연습 관찰 결과
{observations}

## 종합 코칭 리포트를 작성해주세요."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=3000,
        temperature=0.7
    )
    
    return {
        "report": response.choices[0].message.content,
        "model_used": deployment,
        "tokens": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens
        }
    }

def full_analysis_pipeline(video_path=None, use_finetuned=True):
    """전체 분석 파이프라인"""
    print("🎾 테니스 코칭 AI 분석 시작")
    print("=" * 50)
    
    # 1. 프레임 추출 또는 데모 데이터 사용
    if video_path and os.path.exists(video_path):
        from frame_extractor import extract_frames
        print("\n📹 Step 1: 프레임 추출")
        frame_paths = extract_frames(video_path)
        
        # 2. Vision 분석
        print("\n🔍 Step 2: 프레임 분석")
        all_observations = []
        for fp in frame_paths:
            obs = analyze_frame_with_vision(fp)
            all_observations.append(obs)
        observations = "\n\n".join(all_observations)
    else:
        print("\n📋 Step 1-2: 데모 데이터 사용 (영상 없음)")
        demo_frames = get_demo_frames()
        observations = analyze_demo_frames(demo_frames)
    
    print(f"\n관찰 결과:\n{observations[:500]}...\n")
    
    # 3. 코칭 리포트 생성
    print(f"\n📝 Step 3: 코칭 리포트 생성 ({'Fine-tuned' if use_finetuned else 'Base'} 모델)")
    result = generate_coaching_report(observations, use_finetuned)
    
    print(f"\n{'='*50}")
    print(f"📊 분석 완료")
    print(f"모델: {result['model_used']}")
    print(f"토큰 사용: {result['tokens']['total']:,}")
    print(f"{'='*50}")
    print(f"\n{result['report']}")
    
    return result

if __name__ == "__main__":
    # 데모 실행 (영상 없이)
    result = full_analysis_pipeline(use_finetuned=True)
    
    # 결과 저장
    with open("coaching_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n✅ 리포트가 coaching_report.json에 저장되었습니다.")