import os
from pathlib import Path
from dotenv import load_dotenv
from common import measure_time
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from azure.core.credentials import AzureKeyCredential
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid

# ─── 환경 설정 ─────────────────────────────────────────────────────────────
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY      = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOY   = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ─── 클라이언트 초기화 ─────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    model = AZURE_DEPLOY,
    openai_api_version=AZURE_VERSION,  # 필요 시 적절한 버전으로 교체
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
)

# Qdrant 클라이언트 설정
qdrant_client = QdrantClient("localhost", port=6333)

qdrant_client.recreate_collection(
    collection_name="meeting_summaries",
    vectors_config=models.VectorParams(
        size=3072,
        distance=models.Distance.COSINE,
        on_disk=True
    ),
    hnsw_config=models.HnswConfigDiff(
        m=32,
        ef_construct=128,
    ),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    )
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="meeting_summaries",
    embedding=embeddings
)

### Text 파일을 Chunking 한 후 Document List 로 분리합니다.
def chunk_text_to_documents(text_path: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> list:
    ext = Path(text_path).suffix

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    except Exception as e:
        print(f"❌ 텍스트 파일 읽기 실패: {e}")
        raise

    # 텍스트 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(full_text)
    print(f"✅ 텍스트 청킹 완료: {len(chunks)}개 청크 생성")

    # Document 리스트로 변환
    source_name = Path(text_path).stem
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    return docs


# 현재는 FAISS 로 임베딩 중인데 이것을 qdrant로 바꿔주면 됩니다.
@measure_time
def vector_embedding(text_path: str, username: str, openai_api_key: str = None):
    """
    1) 텍스트 파일 청킹
    2) Azure OpenAI로 임베딩
    3) Qdrant에 Upsert

    Args:
        text_path: 텍스트 파일 경로
        username: 사용자 이름
        openai_api_key: OpenAI API 키 (환경변수에서 가져오지 않을 경우)

    Returns:
        생성된 벡터스토어 경로
    """

    # 텍스트 파일 청킹 및 Document 생성
    print(f'🔄 텍스트 파일 청킹 시작: {text_path} ')
    docs = chunk_text_to_documents(text_path)
    texts = [d.page_content for d in docs]

    # 임베딩 및 벡터스토어 생성
    print(f"🔄 벡터 임베딩 시작: {len(docs)}개 문서")
    vector_store.add_documents(docs)

    print(f"✅ 벡터 임베딩 저장 완료")
    return vector_store

if __name__ == "__main__":
    print(vector_embedding("./reference/cleaned_example.txt", "jhkim"))
    # docs = chunk_text_to_documents("./Data/Sound/jhkim/01_transcript.txt")
    # print(docs)
