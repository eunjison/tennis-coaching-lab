# test_connection.py
import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

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

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[
        {"role": "user", "content": "안녕하세요! 테니스에 대해 알고 있나요?"}
    ],
    max_tokens=200
)

print("✅ 연결 성공!")
print(f"응답: {response.choices[0].message.content}")
print(f"토큰 사용량: {response.usage}")