# azure_rag.py
"""
Azure AI Search를 활용한 RAG 검색 모듈
setup_rag.py에서 생성한 인덱스(tennis-knowledge)를 사용합니다.
"""

import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load .env from shared directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from lab08_evaluation
env_path = os.path.join(project_root, 'shared', '.env')
load_dotenv(env_path)


class AzureSearchRAG:
    """Azure AI Search 기반 RAG 검색 엔진"""

    def __init__(self):
        # Azure AI Search 클라이언트
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX", "tennis-knowledge"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
        )

        # OpenAI 클라이언트 (임베딩용) - Azure Entra ID 인증
        credential = DefaultAzureCredential()
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

        # 임베딩 모델 배포명 (.env에 추가 가능)
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
        )

        print(f"✅ Azure AI Search RAG 초기화 완료")
        print(f"   인덱스: {os.getenv('AZURE_SEARCH_INDEX', 'tennis-knowledge')}")

    def _get_embedding(self, text):
        """텍스트 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_deployment,
            input=text,
        )
        return response.data[0].embedding

    def search(self, query, top_k=2):
        """
        하이브리드 검색: 키워드 + 벡터 검색 동시 수행

        Args:
            query: 검색 쿼리 (고객 관찰 결과)
            top_k: 반환할 상위 문서 수

        Returns:
            [{"title": ..., "content": ..., "score": ...}, ...]
        """
        # 쿼리 임베딩 생성
        query_embedding = self._get_embedding(query)

        # 하이브리드 검색 (키워드 + 벡터)
        results = self.search_client.search(
            search_text=query,                      # 키워드 검색
            vector_queries=[
                VectorizableTextQuery(
                    text=query,                     # 벡터 검색 (자동 임베딩)
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ] if False else [],  # VectorizableTextQuery 미지원 시 아래 방식 사용
            top=top_k,
            select=["id", "title", "category", "content"],
        )

        search_results = []
        for result in results:
            search_results.append({
                "title": result["title"],
                "content": result["content"],
                "category": result.get("category", ""),
                "score": result["@search.score"],
            })

        return search_results

    def search_vector(self, query, top_k=2):
        """
        순수 벡터 검색 (시맨틱 유사도 기반)

        setup_rag.py에서 content_vector 필드에 임베딩을 저장했으므로,
        쿼리 임베딩과 비교하여 가장 유사한 문서를 반환합니다.
        """
        from azure.search.documents.models import VectorizedQuery

        query_embedding = self._get_embedding(query)

        results = self.search_client.search(
            search_text=None,  # 키워드 검색 없이 벡터만
            vector_queries=[
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ],
            top=top_k,
            select=["id", "title", "category", "content"],
        )

        search_results = []
        for result in results:
            search_results.append({
                "title": result["title"],
                "content": result["content"],
                "category": result.get("category", ""),
                "score": result["@search.score"],
            })

        return search_results

    def search_hybrid(self, query, top_k=2):
        """
        하이브리드 검색 (키워드 + 벡터 결합)
        가장 정확한 결과를 얻을 수 있는 방식입니다.
        """
        from azure.search.documents.models import VectorizedQuery

        query_embedding = self._get_embedding(query)

        results = self.search_client.search(
            search_text=query,  # 키워드 검색
            vector_queries=[
                VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ],
            top=top_k,
            select=["id", "title", "category", "content"],
        )

        search_results = []
        for result in results:
            search_results.append({
                "title": result["title"],
                "content": result["content"],
                "category": result.get("category", ""),
                "score": result["@search.score"],
            })

        return search_results

    def search_semantic(self, query, top_k=2):
        """
        시맨틱 랭킹 검색 (Azure AI Search의 시맨틱 랭커 활용)
        setup_rag.py에서 SemanticConfiguration을 설정한 경우 사용 가능합니다.
        """
        results = self.search_client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name="my-semantic-config",
            top=top_k,
            select=["id", "title", "category", "content"],
        )

        search_results = []
        for result in results:
            search_results.append({
                "title": result["title"],
                "content": result["content"],
                "category": result.get("category", ""),
                "score": result["@search.score"],
                "reranker_score": result.get("@search.reranker_score", None),
            })

        return search_results


# ── 전역 인스턴스 (rag_evaluation.py에서 호출용) ──

_rag = None

def get_rag():
    """simple_rag.py의 get_rag()와 동일한 인터페이스"""
    global _rag
    if _rag is None:
        _rag = AzureSearchRAG()
    return _rag


# ── 테스트 ──

if __name__ == "__main__":
    rag = get_rag()

    test_query = "포핸드 스트로크 시 라켓을 너무 높이 들어올림, 팔꿈치가 완전히 펴짐"

    print("\n" + "=" * 60)
    print("🔍 검색 테스트")
    print(f"   쿼리: {test_query[:50]}...")
    print("=" * 60)

    # 1) 키워드 검색
    print("\n📌 1. 키워드 검색")
    results = rag.search(test_query, top_k=2)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['title']}")

    # 2) 벡터 검색
    print("\n📌 2. 벡터 검색")
    results = rag.search_vector(test_query, top_k=2)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['title']}")

    # 3) 하이브리드 검색
    print("\n📌 3. 하이브리드 검색 (키워드 + 벡터)")
    results = rag.search_hybrid(test_query, top_k=2)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['title']}")

    print("\n✅ 검색 테스트 완료")