from qdrant_client import QdrantClient
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI 클라이언트 설정
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Qdrant 클라이언트 설정
qdrant_client = QdrantClient("localhost", port=6333)

# 검색할 쿼리 벡터 생성
query = "Jenkins와 CI/CD와 관련된 회의가 있어?"
query_vector = client.embeddings.create(
    model="text-embedding-3-large",
    input=query
).data[0].embedding

# Qdrant에서 검색 실행
results = qdrant_client.search(
    collection_name="meeting_summaries",
    query_vector=query_vector,
    limit=10
)

# 검색된 결과 출력
for hit in results:
    print(hit.payload)
