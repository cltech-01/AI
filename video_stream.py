import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from config import Settings

# 설정 불러오기
settings = Settings()

# 라우터 생성
router = APIRouter()

# 파일 스트리밍을 위한 함수
def iterfile(file_path, chunk_size=1024*1024):
    """
    파일을 청크 단위로 읽어서 반환하는 제너레이터 함수
    chunk_size: 청크 크기 (기본값: 1MB)
    """
    with open(file_path, mode="rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk

# 파일 닫기 함수
def close_file(file_path):
    """백그라운드 태스크로 실행될 파일 닫기 함수"""
    print(f"Streaming completed for: {file_path}")

# 영상 스트리밍 엔드포인트
@router.get("/stream/{user_id}/{video_id}")
async def stream_video(
    user_id: str = Path(..., description="사용자 ID"),
    video_id: str = Path(..., description="비디오 ID (UUID)"),
    ext: Optional[str] = ".mp4"  # 기본 확장자
):
    try:
        # 영상 파일 경로 구성
        video_path = os.path.join(settings.UPLOAD_DIR, user_id, f"{video_id}{ext}")
        
        # 파일 존재 확인
        if not os.path.exists(video_path):
            # 확장자가 지정되지 않았다면 다른 확장자의 파일도 확인
            if ext == ".mp4":
                for possible_ext in [".avi", ".mov", ".mkv", ".webm"]:
                    alt_path = os.path.join(settings.UPLOAD_DIR, user_id, f"{video_id}{possible_ext}")
                    if os.path.exists(alt_path):
                        video_path = alt_path
                        ext = possible_ext
                        break
            
            # 여전히 파일이 없는 경우 404 에러
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail=f"Video file not found: {video_id}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(video_path)
        
        # 미디어 타입 설정
        media_type = None
        if ext == ".mp4":
            media_type = "video/mp4"
        elif ext == ".avi":
            media_type = "video/x-msvideo"
        elif ext == ".mov":
            media_type = "video/quicktime"
        elif ext == ".mkv" or ext == ".webm":
            media_type = "video/webm"
        else:
            media_type = "application/octet-stream"
        
        # 응답 헤더 설정
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600"
        }
        
        # 스트리밍 응답 반환
        return StreamingResponse(
            iterfile(video_path),
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(close_file, video_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming video: {str(e)}")
