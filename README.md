## 요구사항
Python 3.11.8
파이썬 가상환경

```
일단 파이썬 요기 디렉토리만 가상환경 3.11.8 로 세팅하는법..
brew install pyenv
echo 'eval "$(pyenv init --path)"' >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
pyenv install 3.11.8
pyenv local 3.11.8
```

###  1. 가상환경 생성
 - python -m venv venv

### 2. 가상환경 활성화 (macOS/Linux)
- source venv/bin/activate

### 3. 라이브러리 설치
pip install -r requirements.txt


### 4. 시스템 패키지 설치
- brew install ffmpeg


### 4.5 env 파일 생성 
env 파일 생성해야 합니당: 
```

UPLOAD_DIR=./Data/Media
BACKEND_URL=http://localhost:8080
HOST=0.0.0.0
PORT=8000
RELOAD=True
OPENAI_API_KEY=[노션](https://www.notion.so/dong2ast/Secret-1d98d4c0362780bfb9d1d757bf3df23f)
WHISPER_MODEL_NAME=small
```


### 5. API 서버 실행 
- python main.py


** Unit Test 하는법
- python rag.py
- python embedding.py 


