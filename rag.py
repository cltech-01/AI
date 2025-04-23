import os
from dotenv import load_dotenv
from embedding import vector_embedding, vector_store
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

# 환경변수 로딩
load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

class State(TypedDict):
    user_id: int
    text: str
    summary: str
    chunks: list[str]
    documents: list
    status: str


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


from langchain.schema import Document

def extract_keywords(state: State):
    prompt = """
    다음 강의 내용을 보고 중요한 기술 키워드나 주제를 3~5개 추출해주세요.
    단어로만 추출하고, 쉼표로 구분해주세요. 설명은 하지 마세요.

    """

    chunks = state['chunks']
    documents = []

    for chunk in chunks:
        response = llm.invoke(prompt + chunk)
        keywords = [kw.strip() for kw in response.content.split(",")]

        doc = Document(
            page_content=chunk,
            metadata={
                "keywords": keywords,
                "user_id": state["user_id"]
            }
        )

        documents.append(doc)
    
    return {"documents": documents}


def embedding(state: State):
    vector_store.add_documents(state["documents"])

    return state


builder = StateGraph(State)

builder.add_node("split_text_for_filtering", split_text_for_filtering)
builder.add_node("filter_small_talk", filter_small_talk)
builder.add_node("summarize_text", summarize_text)
builder.add_node("split_text_for_embedding", split_text_for_embedding)
builder.add_node("extract_keywords", extract_keywords)
builder.add_node("embedding", embedding)

builder.add_edge(START, "split_text_for_filtering")
builder.add_edge("split_text_for_filtering", "filter_small_talk")
builder.add_edge("filter_small_talk", "summarize_text")
builder.add_edge("summarize_text", "split_text_for_embedding")
builder.add_edge("split_text_for_embedding", "extract_keywords")
builder.add_edge("extract_keywords", "embedding")
builder.add_edge("embedding", END)

graph = builder.compile()


def store_RAG_data(text: str, user_id: int):
    state: State = {
        "text": text,
        "user_id": user_id
    }

    graph.invoke(state)


# QA 시스템 구성
def setup_qa_system(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# 질문 수행 함수
def answer_question(qa_chain, question):
    result = qa_chain({"query": question})
    print(f"질문: {question}\n")
    print(f"답변: {result['result']}\n")
    print("참조 문서:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.page_content} [{doc.metadata}]")
    print("\n" + "-"*50 + "\n")

# 실행
if __name__ == "__main__":
    # print("📦 벡터 임베딩 + VectorStore 로딩...")
    # vectorstore = vector_embedding("reference/cleaned_example.txt", "jhkim")

    print("데이터 전처리 & 임베딩 중...")
    # TODO: text data, user_id parameter로 받기
    test_text = ""
    with open("./reference/cleaned_example.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    store_RAG_data(test_text, 1)
    print("완료!")

    print("🔧 QA 시스템 구성 중...")
    qa_chain = setup_qa_system(vector_store)

    print("📤 질문 보내기...")
    answer_question(qa_chain, "KT에서 CI/CD는 어떻게 처리해?")