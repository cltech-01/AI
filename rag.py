from dotenv import load_dotenv
from embedding import vector_embedding
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

def setup_qa_system(vectorstore):
    """
    벡터 스토어를 사용하여 QA 시스템을 설정합니다.
    
    Args:
        vectorstore: 임베딩된 벡터 스토어
    
    Returns:
        RetrievalQA 체인
    """
    # MMR 검색 방식의 Retriever 설정
    retriever = vectorstore.as_retriever(search_type="mmr")
    retriever.search_kwargs.update({
        "k": 10,
        "fetch_k": 100,
        "maximal_marginal_relevance": True
    })
    
    # QA 체인 구성
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa

def ask_question(qa, query):
    """
    질문에 대한 답변을 생성합니다.
    
    Args:
        qa: QA 체인
        query: 질문 문자열
    
    Returns:
        답변 결과
    """
    print(f"\n💬 질문: {query}")
    result = qa(query)
    
    print(f"🧠 답변: {result['result']}")
    print("\n📄 참고 문서:")
    for i, doc in enumerate(result['source_documents']):
        print(f"\n문서 {i+1}:")
        print(f"내용: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"내용: {doc.page_content}")
        print(f"메타데이터: {doc.metadata}")
    
    return result

if __name__ == "__main__":
    load_dotenv()
    vectorstore = vector_embedding("reference/cleaned_example.txt", "jhkim")
    qa = setup_qa_system(vectorstore)
    ask_question(qa, "KT에서 CI/CD는 어떻게 처리해?")

