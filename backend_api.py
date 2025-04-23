import requests
from config import settings

# 백엔드 엔티티 생성 API 호출
# task_id : file의 uuid임 
import requests
from config import settings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def create_backend_entity(lecture_uuid: str, user_id: str, original_filename: str, status: str, backend_url:str = os.getenv("BACKEND_URL")):
    try:
        # 백엔드 API 엔드포인트 (Postman에서 쓰는 URL 기준)
        api_url = f"{backend_url}/lectures"  # 예: http://localhost:8000/lectures

        # Postman에서 사용한 필드명 기준으로 수정
        data = {
            "fileName": original_filename,
            "uuid": lecture_uuid,
            "username": user_id,
            "status": status
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(api_url, json=data, headers=headers)

        print(f"✅ 백엔드 응답 코드: {response.status_code}")
        print(f"📦 백엔드 응답 내용: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ 백엔티티 생성 실패: {e}")
        return None
    

# 백엔드 상태 업데이트 콜백
def notify_backend(lecture_uuid: str, status: str):
    try:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "status": status
        }

        # 실제 API 호출 (백엔드가 없으므로 주석 처리)
        callback_url = f"{os.getenv('BACKEND_URL')}/lectures/{lecture_uuid}"
        res = requests.patch(callback_url, json=data)
        
        print(f"✅ 상태 보고 (시뮬레이션): {status}")
        
    except Exception as e:
        print(f"❌ 상태 보고 실패: {e}")

def send_summary_to_backend(lecture_uuid: str, summary: str):
    """
    백엔드에 요약본 전송
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": summary
        }

        url = f"{os.getenv('BACKEND_URL')}/lectures/{lecture_uuid}/summary"
        response = requests.patch(url, json=data, headers=headers)

        print(f"✅ 요약 전송 완료: {response.status_code}")
        print(f"📦 응답 내용: {response.text}")

    except Exception as e:
        print(f"❌ 요약 전송 실패: {e}")


def send_cleantext_to_backend(lecture_uuid: str, clean_text: str):
    """
    백엔드에 요약본 전송
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "fullText": clean_text
        }

        url = f"{os.getenv('BACKEND_URL')}/lectures/{lecture_uuid}"
        response = requests.patch(url, json=data, headers=headers)
        
        print(f"✅ 요약 전송 완료: {response.status_code}")
        print(f"📦 응답 내용: {response.text}")

    except Exception as e:
        print(f"❌ 요약 전송 실패: {e}")

if __name__ == "__main__":
    pass