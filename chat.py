import os
import uuid
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, Body, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import json
import asyncio
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA

# 채팅 요청 모델
class ChatRequest(BaseModel):
    userId: str
    message: str
    lectureId: Optional[str] = None
    conversationId: Optional[str] = None


# 스트리밍 콜백 핸들러
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.queue = asyncio.Queue()
    
    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put(token)
        self.tokens.append(token)


# FastAPI 라우터 생성
router = APIRouter()

# 채팅 엔드포인트
@router.post("/chat")
async def chat(request: ChatRequest = Body(...)):
    try:
        # 요청 데이터 추출
        user_id = request.userId
        message = request.message
        lecture_id = request.lectureId
        conversation_id = request.conversationId or str(uuid.uuid4())
        
        # 스트리밍 응답 반환
        return StreamingResponse(
            stream_chat_response(user_id, message, lecture_id, conversation_id),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })

async def stream_chat_response(user_id, message, lecture_id, conversation_id) -> AsyncGenerator[str, None]:
    # 스트리밍 핸들러 생성
    streaming_handler = StreamingCallbackHandler()
    
    try:
        # 초기 응답 생성
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
        
        # Qdrant 클라이언트 및 임베딩 모델 설정
        qdrant_client = QdrantClient("localhost", port=6333)
        
        # Azure OpenAI 설정
        streaming_llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            streaming=True,
            callbacks=[streaming_handler]
        )
        
        # 컬렉션 이름 결정
        collection_name = "meeting_summaries"
        
        # Azure OpenAI 임베딩 설정
        from langchain_openai import AzureOpenAIEmbeddings
        
        embeddings = AzureOpenAIEmbeddings(
            model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
        )
        
        # 해당 컬렉션의 벡터스토어 생성
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings
        )
        
        # 리트리버 설정
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # QA 체인 설정
        qa_chain = RetrievalQA.from_chain_type(
            llm=streaming_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # 비동기로 질문 처리 시작
        task = asyncio.create_task(
            asyncio.to_thread(
                qa_chain,
                {"query": message}
            )
        )
        
        # 토큰 스트리밍
        while not task.done():
            try:
                # 토큰 대기 (0.1초 타임아웃)
                token = await asyncio.wait_for(streaming_handler.queue.get(), timeout=0.1)
                yield f"data: {json.dumps({'type': 'chunk', 'content': token}, ensure_ascii=False)}\n\n"
                streaming_handler.queue.task_done()
            except asyncio.TimeoutError:
                # 타임아웃이면 다시 대기
                continue
        
        # 작업 결과 가져오기
        result = await task
        
        # 참조 문서 정보 전송
        sources = []
        for i, doc in enumerate(result["source_documents"]):
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
        
        # 스트림 종료
        yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        # 오류 발생 시
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


