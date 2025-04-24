import os
from pathlib import Path
from dotenv import load_dotenv
from common import measure_time
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from backend_api import send_summary_to_backend, send_cleantext_to_backend
from langchain_text_splitters.konlpy import KonlpyTextSplitter
from qdrant_client.http.exceptions import UnexpectedResponse


from typing import List
# ─── 환경 설정 ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

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
    temperature=0.1
)



# ─── 클라이언트 초기화 ─────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    model = AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,  # 필요 시 적절한 버전으로 교체
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT
)

# state 정의
class State(TypedDict):
    user_id: int
    text: str
    summary: str
    chunks: list[str]
    documents: list
    lecture_uuid: str
    cleaned_text: str



# Qdrant 클라이언트 설정
qdrant_client = QdrantClient("localhost", port=6333)


try:
    qdrant_client.create_collection(
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
    print("✅ Qdrant 컬렉션 생성 완료.")
except UnexpectedResponse as e:
    if "already exists" in str(e):
        print("⚠️ Qdrant 컬렉션이 이미 존재합니다. 생성을 건너뜁니다.")
    else:
        raise e
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="meeting_summaries",
    embedding=embeddings
)

@measure_time
def split_text_for_filtering(state: State):
    splitter = KonlpyTextSplitter(chunk_size=2000, chunk_overlap=0)
    # splitter = KoreanSentenceSplitter(chunk_size=2000)

    return {"chunks": splitter.split_text(state["text"])}

# @measure_time
# def split_text_for_filtering(state: State):
#     text_splitter_for_filtering = RecursiveCharacterTextSplitter(
#         chunk_size=2000, # gpt가 효율적으로 처리 가능한 글자수
#         chunk_overlap=50,
#     )

#     chunks = text_splitter_for_filtering.split_text(state['text'])

    return {"chunks": chunks}

@measure_time
def refine_to_natural_korean(state: State):
    prompt = """
다음은 음성 인식(STT)을 통해 변환된 문장입니다. 아래 텍스트를 자연스러운 한국어로 고쳐주세요. 요약하지 말고 어색한 부분만 수정해주세요.
또한 LLM(AI)이 생성한 듯한 인위적인 표현, 과한 말투, 또는 markdown 형식의 강조(*기울임*) 등을 제거해주세요.
순수한 텍스트처럼 읽히도록 고쳐주세요. 단, 내용을 요약하거나 삭제하지 말고 자연스럽게 정제만 해주세요.절대로 원문의 내용을 변형시켜서는 안됩니다.
추가로 대답할떄, 대답을 하지 않고 원문을 그대로 남겨주세요.

예시:
원문: 또 한편 이런 왜 분야를 넘나드는 지식을 잡하기라고 부르고 그 앞에 쓸데없다는 쓰시고가 붙어있을까.
수정: 또 한편, 왜 이런 분야를 넘나드는 지식을 잡학이라고 부르고, 그 앞에 쓸데없다는 수식어가 붙어 있을까.

예시:
원문: 이어 제러드 다이아몬드의 저서 *총균쇠*가 소개됩니다.
수정: 이어 제러드 다이아몬드의 저서 총균쇠가 소개됩니다.

"""
    result = ""
    for chunk in state["chunks"]:
        response = llm.invoke(prompt + chunk)
        result += response.content if response.content else chunk

    send_cleantext_to_backend(state["lecture_uuid"], result)
    return {
        "text": result,         # 이후 작업용 텍스트 (summarize 등에 사용)
        "chunks": [],           # 이후 다시 split 예정
        "cleaned_text": result  # 정제된 결과 별도 보관
}


@measure_time
def summarize_text(state: State):
    prompt = f"""
다음은 IT 기술 관련 강의 내용입니다.  
전체 내용을 한눈에 파악할 수 있도록 핵심 개념, 주요 흐름, 설명 포인트를 중심으로 **간결하고 명확하게** 정리해주세요.  
불필요한 예시나 수사는 제외하고, **1000자 이내로 요약**해주세요.

강의 내용:
    {state['cleaned_text']}
    """
    response = llm.invoke(prompt)
    
    # TODO: 백엔드 API 호출
    # print(state['text'])
    # print('hi\n ', state,'\nhi')
    lecture_uuid = state.get("lecture_uuid")
    # print(lecture_uuid)
    if lecture_uuid:
        send_summary_to_backend(lecture_uuid, response.content)
        # print(response.content)
    # 백엔드 API 호출
    return {"summary": response.content}

@measure_time
def split_text_for_embedding(state: State):
    splitter = KonlpyTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    # splitter = KoreanSentenceSplitter(chunk_size=1000)

    chunks = splitter.split_text(state['cleaned_text'])

    return {"chunks": chunks}

@measure_time
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
                "keywords": keywords,
                "lecture_uuid": state["lecture_uuid"]
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


builder.add_node("split_text_for_filtering", split_text_for_filtering) ## AI 가 이해하기 쉽도록 2000 한국어 단위로  끊습니다.
builder.add_node("refine_to_natural_korean", refine_to_natural_korean) ## 자연스러운 한국어로 변환합니다. [MarkDown 언어, 구어체 수정]
builder.add_node("summarize_text", summarize_text) # 여기서 텍스트를 합치는 pipeline 이 필요합니다.  summarize는 아래로 옮겨야함
builder.add_node("split_text_for_embedding", split_text_for_embedding) # 1000개식 청킹후 embedding을 합니다. 
builder.add_node("extract_keywords", extract_keywords) # 각 document 별로 키워드를 추출합니다. 
builder.add_node("vector_embedding", vector_embedding) # 이후 벡터 임베딩을 합니다. 
                                                       # 이후 AI에게 Summary 요청을 하고 백엔드에 API 요청을 합니다. node 필요 


builder.add_edge(START, "split_text_for_filtering")
builder.add_edge("split_text_for_filtering", "refine_to_natural_korean")
builder.add_edge("refine_to_natural_korean", "summarize_text")
builder.add_edge("summarize_text", "split_text_for_embedding")
builder.add_edge("split_text_for_embedding", "extract_keywords")
builder.add_edge("extract_keywords", "vector_embedding")
builder.add_edge("vector_embedding", END)

graph = builder.compile()


# main에서 쓰는 method
@measure_time
def store_data(path: str, user_id: int, lecture_uuid: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"텍스트 파일 읽기 실패: {e}")
        raise

    state: State = {
        "text": text,
        "user_id": user_id,
        "lecture_uuid": lecture_uuid
    }
    # print(state)

    graph.invoke(state)


if __name__ == "__main__":
    store_data("./reference/cleaned_example.txt", 1,"12")
