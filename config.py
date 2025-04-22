from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 기존 설정
    UPLOAD_DIR: str
    BACKEND_URL: str
    WHISPER_MODEL_NAME: str

    # Azure OpenAI 관련 설정 추가
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str

    OPENAI_API_KEY: str

    # 기타
    API_URL: str

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
