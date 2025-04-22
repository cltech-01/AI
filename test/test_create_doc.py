import requests

BASE_URL = "http://localhost:8000"

def test_create_document():
    """
    /documents 에 JSON 형태로 문서 단건 upsert
    """
    document = {
        "title": "회의 요약 1",
        "department": "IT",
        "keywords": ["인프라", "프로젝트", "API"],
        # content 길이가 너무 길면 청킹 시 자동 분할됩니다.
        "content": open("01_transcribed.txt", "r", encoding="utf-8").read()[:8000]
    }
    resp = requests.post(f"{BASE_URL}/documents", json=document)
    print("CREATE_DOCUMENT", resp.status_code, resp.json())


def test_ingest_file():
    """
    /ingest-file 로 텍스트 파일 전체 업로드해서 청킹→임베딩→Qdrant upsert
    """
    with open("01_transcribed.txt", "rb") as fp:
        files = {
            "file": ("01_transcribed.txt", fp, "text/plain")
        }
        # source: metadata 에 들어갈 키(선택)
        data = {"source": "회의록1"}
        resp = requests.post(f"{BASE_URL}/ingest-file", files=files, data=data)
    print("INGEST_FILE", resp.status_code, resp.json())

if __name__ == "__main__":
    print("1) create_document")
    test_create_document()
    print("\n2) ingest_file")
    test_ingest_file()