# simple_rag.py (Azure AI Search 대신 로컬 RAG)
import numpy as np
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from tennis_knowledge import get_knowledge_docs
import os
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

class SimpleRAG:
    def __init__(self):
        self.docs = get_knowledge_docs()
        self.embeddings = []
        self._build_index()
    
    def _build_index(self):
        """문서 임베딩 생성"""
        print("📚 RAG 인덱스 구축 중...")
        for doc in self.docs:
            response = client.embeddings.create(
                model="text-embedding-3-small",  # 배포명에 맞게 수정
                input=doc["content"][:8000]
            )
            self.embeddings.append(response.data[0].embedding)
        print(f"✅ {len(self.docs)}개 문서 인덱싱 완료")
    
    def search(self, query, top_k=2):
        """코사인 유사도 기반 검색"""
        query_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_emb = np.array(query_response.data[0].embedding)
        
        scores = []
        for emb in self.embeddings:
            emb_arr = np.array(emb)
            similarity = np.dot(query_emb, emb_arr) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb_arr)
            )
            scores.append(similarity)
        
        # 상위 K개 문서 반환
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "title": self.docs[idx]["title"],
                "content": self.docs[idx]["content"],
                "score": float(scores[idx])
            })
        return results

# 전역 인스턴스
rag = None

def get_rag():
    global rag
    if rag is None:
        rag = SimpleRAG()
    return rag