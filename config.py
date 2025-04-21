from pydantic_settings import BaseSettings  # ✅ 변경된 import 경로

class Settings(BaseSettings):
    # 기존 설정
    UPLOAD_DIR: str
    CALLBACK_URL: str
    
    # 서버 설정 추가
    HOST: str = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 가능
    PORT: int = 8000
    RELOAD: bool = True

    class Config:
        env_file = ".env"