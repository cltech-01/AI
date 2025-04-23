from pydantic_settings import BaseSettings  # ✅ 변경된 import 경로

class Settings(BaseSettings):
    # 기존 설정
    UPLOAD_DIR: str
    BACKEND_URL: str

    WHISPER_MODEL_NAME: str

    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str

    OPENAI_API_KEY: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_OPENAI_EMBEDDING_API_KEY: str
    AZURE_OPENAI_EMBEDDING_ENDPOINT: str
    AZURE_OPENAI_EMBEDDING_API_VERSION: str

    # 서버 설정 추가
    HOST: str = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 가능
    PORT: int = 8001
    RELOAD: bool = True

    model_config = {  # class Config 대신 model_config 사용
        "env_file": ".env",
        "extra": "ignore"  # 정의되지 않은 추가 필드 무시
    }

settings = Settings()