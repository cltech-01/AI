import os
import uuid
import shutil
import requests
import threading
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from config import Settings

settings = Settings()
app = FastAPI()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    user_key: str = Form(...),
):
    try:
        # 사용자 디렉토리 생성
        user_dir = os.path.join(settings.UPLOAD_DIR, user_key)
        os.makedirs(user_dir, exist_ok=True)

        # 확장자 추출
        ext = os.path.splitext(file.filename)[1]

        # 중복 없는 랜덤 파일명 생성
        while True:
            random_name = str(uuid.uuid4())
            save_path = os.path.join(user_dir, f"{random_name}{ext}")
            if not os.path.exists(save_path):
                break

        # 파일 저장
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        

        print(f"✅ 영상 저장 완료: {save_path}")

        # 콜백
        lecture_id = random_name
        notify_backend(lecture_id, "1. 영상 업로드 중", user_key)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "filename": f"{random_name}{ext}",
            "path": save_path
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })

def notify_backend(lecture_id: str, status: str, user_id: str):
    try:
        res = requests.post(settings.CALLBACK_URL, json={
            "userId": user_id,
            "lectureId": lecture_id,
            "status": status
        })
        print(f"✅ 상태 보고됨: {status}")
    except Exception as e:
        print(f"❌ 상태 보고 실패: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.RELOAD
    )