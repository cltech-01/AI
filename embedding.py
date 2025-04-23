import os
from pathlib import Path
from dotenv import load_dotenv
from common import measure_time
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ─── 환경 설정 ─────────────────────────────────────────────────────────────
load_dotenv()

AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_API_KEY      = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME   = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_API_VERSION  = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview")

# ─── OpenAI 초기화 ─────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)



# ─── 클라이언트 초기화 ─────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,  # 필요 시 적절한 버전으로 교체
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
)

# state 정의
class State(TypedDict):
    user_id: int
    text: str
    summary: str
    chunks: list[str]
    documents: list

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

# ### Text 파일을 Chunking 한 후 Document List 로 분리합니다.
# def chunk_text_to_documents(text_path: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> list:
#     ext = Path(text_path).suffix

#     try:
#         with open(text_path, "r", encoding="utf-8") as f:
#             full_text = f.read()
#     except Exception as e:
#         print(f"❌ 텍스트 파일 읽기 실패: {e}")
#         raise

#     # 텍스트 청킹
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )

#     chunks = text_splitter.split_text(full_text)
#     print(f"✅ 텍스트 청킹 완료: {len(chunks)}개 청크 생성")

#     # Document 리스트로 변환
#     source_name = Path(text_path).stem
#     docs = [
#         Document(
#             page_content=chunk,
#             metadata={
#                 "source": source_name,
#                 "chunk_index": i
#             }
#         )
#         for i, chunk in enumerate(chunks)
#     ]

#     return docs


def split_text_for_filtering(state: State):
    text_splitter_for_filtering = RecursiveCharacterTextSplitter(
        chunk_size=2000, # gpt가 효율적으로 처리 가능한 글자수
        chunk_overlap=50,
    )

    chunks = text_splitter_for_filtering.split_text(state['text'])

    return {"chunks": chunks}


def filter_small_talk(state: State):
    prompt = """
    다음은 IT 기술 강의입니다. 다음 내용에서 IT 기술과 관련없는 일상적인 대화나 불필요한 감탄사/웃음 등은 제거해주세요. 제거 후 요약하지 말고 원문 그대로 남겨주세요.:

    """
    result = ""
    chunks = state['chunks']

    for chunk in chunks:
        response = llm.invoke(prompt + chunk)

        if response.content:
            result += response.content
        else:
            result += chunk

    return {"text": result}


def summarize_text(state: State):
    prompt = f"""
    다음은 IT 기술 강의입니다. 강의를 1000자 내외로 요약해주세요.:

    {state['text']}
    """
    response = llm.invoke(prompt)
    
    # TODO: 백엔드 API 호출
    return {"summary": response.content}


def split_text_for_embedding(state: State):
    text_splitter_for_embedding = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )

    chunks = text_splitter_for_embedding.split_text(state['text'])

    return {"chunks": chunks}


def extract_keywords(state: State):
    prompt = """
    다음 강의 내용을 보고 중요한 기술 키워드나 주제를 3~5개 추출해주세요.
    단어로만 추출하고, 쉼표로 구분해주세요. 설명은 하지 마세요.

    """

    chunks = state['chunks']
    documents = []

    for i, chunk in enumerate(chunks):
        response = llm.invoke(prompt + chunk)
        keywords = [kw.strip() for kw in response.content.split(",")]

        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_index": i,
                "user_id": state["user_id"],
                "keywords": keywords
            }
        )

        documents.append(doc)
    
    return {"documents": documents}


@measure_time
def vector_embedding(state: State):
    vector_store.add_documents(state["documents"])

    return state


# ─── LangGraph build ─────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("split_text_for_filtering", split_text_for_filtering)
builder.add_node("filter_small_talk", filter_small_talk)
builder.add_node("summarize_text", summarize_text)
builder.add_node("split_text_for_embedding", split_text_for_embedding)
builder.add_node("extract_keywords", extract_keywords)
builder.add_node("vector_embedding", vector_embedding)

builder.add_edge(START, "split_text_for_filtering")
builder.add_edge("split_text_for_filtering", "filter_small_talk")
builder.add_edge("filter_small_talk", "summarize_text")
builder.add_edge("summarize_text", "split_text_for_embedding")
builder.add_edge("split_text_for_embedding", "extract_keywords")
builder.add_edge("extract_keywords", "vector_embedding")
builder.add_edge("vector_embedding", END)

graph = builder.compile()


# main에서 쓰는 method
@measure_time
def store_data(path: str, user_id: int):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"텍스트 파일 읽기 실패: {e}")
        raise

    state: State = {
        "text": text,
        "user_id": user_id
    }

    graph.invoke(state)


if __name__ == "__main__":
    store_data("./reference/cleaned_example.txt", 1)