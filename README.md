# FastAPI Demo

A simple FastAPI application with basic CRUD operations.

## 설치 방법

1. 필요한 패키지 설치하기:
```
pip install -r requirements.txt
```

## 실행 방법

1. 서버 실행하기:
```
python main.py
```

2. 또는 uvicorn으로 직접 실행:
```
uvicorn main:app --reload
```

## API 문서

서버가 실행되면 아래 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 