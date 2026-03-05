# run_finetuning.py
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

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

# 1. 학습 데이터 업로드
print("📤 학습 데이터 업로드 중...")
training_file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)
print(f"  Training file ID: {training_file.id}")

# 2. 검증 데이터 업로드
print("📤 검증 데이터 업로드 중...")
validation_file = client.files.create(
    file=open("validation_data.jsonl", "rb"),
    purpose="fine-tune"
)
print(f"  Validation file ID: {validation_file.id}")

# 3. Fine-tuning 작업 생성
print("\n🚀 Fine-tuning 시작...")
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-nano-2025-04-14",  # 모델 이름 확인 필요
    hyperparameters={
        "n_epochs": 3,
        # batch_size와 learning_rate_multiplier는 auto 권장
    },
    suffix="tennis-coach"  # 모델 이름에 접미사 추가
)

print(f"  Job ID: {job.id}")
print(f"  Status: {job.status}")
print(f"\n⏳ 학습이 진행 중입니다. 아래 명령어로 상태를 확인하세요:")
print(f"  python check_finetuning_status.py {job.id}")