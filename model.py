from pydantic import BaseModel
from typing import Optional
from typing_extensions import TypedDict

# 채팅 요청 모델
class ChatRequest(BaseModel):
    user_id: str
    message: str
    lecture_id: Optional[str] = None
    conversation_id: Optional[str] = None


class State(TypedDict):
    user_id: int
    text: str
    summary: str
    chunks: list[str]
    documents: list
    status: str
    lecture_uuid: str
