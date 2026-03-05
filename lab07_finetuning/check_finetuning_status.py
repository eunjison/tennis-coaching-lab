# check_finetuning_status.py
import os
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

# Load .env from shared directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab07_finetuning
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)

# Use Azure Entra ID authentication instead of API key
credential = DefaultAzureCredential()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

job_id = sys.argv[1] if len(sys.argv) > 1 else input("Job ID를 입력하세요: ")

job = client.fine_tuning.jobs.retrieve(job_id)

print(f"\n📊 Fine-tuning 작업 상태")
print(f"{'='*50}")
print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
print(f"Model: {job.model}")
print(f"Created: {job.created_at}")

if job.status == "succeeded":
    print(f"\n✅ 학습 완료!")
    print(f"Fine-tuned Model: {job.fine_tuned_model}")
    print(f"\n다음 단계:")
    print(f"1. Azure AI Foundry에서 fine-tuned 모델을 배포하세요.")
    print(f"2. 배포 이름을 .env에 추가하세요:")
    print(f"   AZURE_OPENAI_FT_DEPLOYMENT=<배포이름>")
elif job.status == "failed":
    print(f"\n❌ 학습 실패")
    print(f"Error: {job.error}")
elif job.status == "running":
    # 이벤트 확인
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
    print(f"\n📋 최근 이벤트:")
    for event in events.data:
        print(f"  [{event.created_at}] {event.message}")
else:
    print(f"\n⏳ 현재 상태: {job.status}")