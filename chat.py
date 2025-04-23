import os
import uuid
from typing import Optional, AsyncGenerator
from fastapi import Body, APIRouter
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
from embedding import embeddings

load_dotenv()

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


# 평가 LLM (별도로 생성, 답변 적합성 평가용)
def answer_is_grounded(answer: str, sources: list, eval_llm) -> bool:
    context = "\n".join([doc.page_content for doc in sources])
    eval_prompt = f"""
    아래는 사용자 질문에 대한 AI의 답변과, 그 답변을 뒷받침하는 참고 문서들입니다.

    [AI 답변]
    {answer}

    [참고 문서]
    {context}

    위 답변이 '참고 문서'에 명확하게 근거를 두고 있다면 "yes", 부족하다면 "no"로만 답해주세요.
    """

    eval_result = eval_llm.invoke(eval_prompt)
    decision = eval_result.content.strip().lower()
    return decision == "yes"


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
    # Qdrant 클라이언트 및 임베딩 모델 설정
    qdrant_client = QdrantClient("localhost", port=6333)
    
    # 컬렉션 이름 결정
    collection_name = f"lecture_{lecture_id}" if lecture_id else f"{user_id}_collection"
    
    # 컬렉션 존재 확인
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        # 컬렉션이 없으면 기본 컬렉션 fallback
        collection_name = "meeting_summaries"
        if collection_name not in collection_names:
            yield f"data: {json.dumps({'type': 'chunk', 'content': '컬렉션을 찾을 수 없습니다. 데이터를 먼저 업로드해주세요.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
            return
    
    # 해당 컬렉션의 벡터스토어 생성
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    try:
        # 초기 응답 생성
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
        
        # (1~N차 시도: "비스트림" LLM으로 내부적으로 답변 생성/평가)
        eval_llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            streaming=False,
        )
        
        # Azure OpenAI 설정
        streaming_llm = AzureChatOpenAI(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            streaming=True,
            callbacks=[]
        )
        
        # 리트리버 설정
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Agentic 루프 (최대 2회까지만 재시도, 무한루프 방지)
        max_attempts = 2
        attempt = 0
        answer_is_valid = False
        result = None

        while attempt <= max_attempts and not answer_is_valid:
            # QA 체인 설정
            qa_chain = RetrievalQA.from_chain_type(
                llm=eval_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            # 여긴 스트리밍 안 하고, 그냥 답변만 받음
            result = await asyncio.to_thread(qa_chain, {"query": message})
            sources = result["source_documents"]
            answer_is_valid = answer_is_grounded(result["result"], sources, eval_llm)

            if not answer_is_valid:
                attempt += 1
            else:
                break

        # 스트리밍 핸들러 준비
        streaming_handler = StreamingCallbackHandler()
        streaming_llm.callbacks = [streaming_handler]

        # 컨텍스트(근거 문서) 재사용해서 스트리밍 LLM에 전달
        context = "\n".join([doc.page_content for doc in result["source_documents"]])
        streaming_prompt = f"""
        아래 참고 문서를 바탕으로 질문에 답변해주세요.
        [참고 문서]
        {context}

        질문: {message}
        도움되는 답변:
        """
        
        ## 비동기로 질문 처리 시작
        task = asyncio.create_task(
            asyncio.to_thread(streaming_llm.invoke, streaming_prompt)
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
        await task
        
        # 참조 문서 정보 전송
        sources_json = []
        for i, doc in enumerate(result["source_documents"]):
            sources_json.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_json}, ensure_ascii=False)}\n\n"
        
        # 스트림 종료
        yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        # 오류 발생 시
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


