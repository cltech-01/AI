import os
import uuid
import shutil
from config import Settings
from stt import process_audio
from embedding import vector_embedding
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from backend_api import create_backend_entity, notify_backend
from data_processing import extract_audio_from_video, clean_text



settings = Settings()
app = FastAPI()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# 백그라운드 작업을 위한 별도 함수 정의
def process_video_background(save_path, user_key, random_name, filename):
    try:
        lecture_id = random_name
        # 2. ffmpeg으로 오디오 추출중  
        notify_backend(lecture_id, "오디오 추출중", user_key)
        audio_path = extract_audio_from_video(save_path, user_key, random_name)
        
        # 3. 오디오 stt로 변환 
        notify_backend(lecture_id, "오디오 stt 변환중", user_key)
        _, text_path = process_audio(audio_path, settings.WHISPER_MODEL_NAME)

        # 4. Text 파일 정제 
        notify_backend(lecture_id, "Text 파일 정제중", user_key)
        cleaned_text_path = clean_text(text_path, user_key)

        # 5. Text 파일 vector Embedding 시작 
        notify_backend(lecture_id, "Text 파일 vector Embedding 시작", user_key)
        vector_embedding(cleaned_text_path, user_key)

        notify_backend(lecture_id, "완료", user_key)
    except Exception as e:
        notify_backend(lecture_id, f"오류: {str(e)}", user_key)
        print(f"❌ 처리 중 오류 발생: {e}")

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,  # 추가
    file: UploadFile = File(...),
    user_key: str = Form(...),
):
    try:
        # 프론트 -> AI 서버로 영상 보내는 부분 
        user_dir = os.path.join(settings.UPLOAD_DIR, user_key)
        os.makedirs(user_dir, exist_ok=True)
        ext = os.path.splitext(file.filename)[1]
        while True:
            random_name = str(uuid.uuid4())
            save_path = os.path.join(user_dir, f"{random_name}{ext}")
            if not os.path.exists(save_path):
                break
        
        # 1. 백엔드에다가 영상 엔티티 생성 요청
        create_backend_entity(random_name, user_key, file.filename, "영상 업로드중")
        with open(save_path, "wb") as f: 
            shutil.copyfileobj(file.file, f)
        print(f"✅ 영상 저장 완료: {save_path}")

        # 나머지 작업을 백그라운드에서 실행
        background_tasks.add_task(
            process_video_background, 
            save_path, 
            user_key, 
            random_name,
            file.filename
        )

        # 프론트엔드에 즉시 응답
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "영상이 성공적으로 업로드되었습니다. 처리가 백그라운드에서 진행됩니다.",
            "task_id": random_name  # 클라이언트가 상태를 추적할 수 있는 ID
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.RELOAD
    )