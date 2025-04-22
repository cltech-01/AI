import os
from dotenv import load_dotenv
from embedding import vector_embedding
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

# 🔧 환경변수 로딩
load_dotenv()

# 🔧 QA 시스템 구성
def setup_qa_system(vector_store):
    llm = AzureChatOpenAI(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    # llm = AzureChatOpenAI(
    #         azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    #         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    #         temperature=0.0,
    #     )
    # llm= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# 🔧 질문 수행 함수
def answer_question(qa_chain, question):
    result = qa_chain({"query": question})
    print(f"질문: {question}\n")
    print(f"답변: {result['result']}\n")
    print("참조 문서:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.page_content} [{doc.metadata}]")
    print("\n" + "-"*50 + "\n")

# 🔧 실행
if __name__ == "__main__":
    print("📦 벡터 임베딩 + VectorStore 로딩...")
    vectorstore = vector_embedding("reference/cleaned_example.txt", "jhkim")

    print("🔧 QA 시스템 구성 중...")
    qa_chain = setup_qa_system(vectorstore)

    print("📤 질문 보내기...")
    answer_question(qa_chain, "KT에서 CI/CD는 어떻게 처리해?")