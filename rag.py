from embedding import vector_store, llm
from langchain.chains import RetrievalQA


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
    print("🔧 QA 시스템 구성 중...")
    qa_chain = setup_qa_system(vector_store)

    print("📤 질문 보내기...")
    answer_question(qa_chain, "KT에서 CI/CD는 어떻게 처리해?")