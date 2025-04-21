import os
from pathlib import Path
from dotenv import load_dotenv
from common import measure_time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

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
    텍스트 파일을 벡터로 임베딩하고 FAISS 인덱스를 생성합니다.
    
    Args:
        text_path: 텍스트 파일 경로
        username: 사용자 이름
        openai_api_key: OpenAI API 키 (환경변수에서 가져오지 않을 경우)
        
    Returns:
        생성된 벡터스토어 경로
    """
    # API 키 설정
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API 키가 필요합니다. 환경변수 또는 매개변수로 제공해주세요.")
    
    # 텍스트 파일 청킹 및 Document 생성
    print(f"🔄 텍스트 파일 청킹 시작: {text_path}")
    docs = chunk_text_to_documents(text_path)
    
    # 임베딩 및 벡터스토어 생성
    print(f"🔄 벡터 임베딩 시작: {len(docs)}개 문서")
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    
    # 벡터스토어 저장 # 실제로는 없어도 댐
    # file_stem = Path(text_path).stem
    # embedding_dir = Path(f"./Data/Embeddings/{username}")
    # embedding_dir.mkdir(exist_ok=True, parents=True)
    
    # embedding_path = f"{embedding_dir}/{file_stem}"
    # vectorstore.save_local(embedding_path)
    
    print(f"✅ 벡터 임베딩 저장 완료")
    return vectorstore





if __name__ == "__main__":
    print(vector_embedding("./Data/Text/jhkim/01_transcript.txt", "jhkim"))
    # docs = chunk_text_to_documents("./Data/Sound/jhkim/01_transcript.txt")
    # print(docs)