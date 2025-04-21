import requests
from config import settings

# 백엔드 엔티티 생성 API 호출
# task_id : file의 uuid임 
def create_backend_entity(task_id: str, user_id: str, original_filename: str, status: str):
    try:
        # 백엔드 API 엔드포인트 (실제 운영 시 설정 필요)
        api_url = f"{settings.BACKEND_URL}/api/videos"
        
        # API 요청 데이터
        data = {
            "taskId": task_id,
            "userId": user_id,
            "originalFilename": original_filename,
            "status": status
        }
        
        # API 호출 (현재 백엔드가 없으므로 주석 처리)
        # response = requests.post(api_url, json=data)
        # return response.json()
        
        # 백엔드가 없는 상황을 가정한 로그
        print(f"✅ 백엔드 엔티티 생성 요청 (시뮬레이션): {data}")
        return {"id": task_id, "status": "created"}
        
    except Exception as e:
        print(f"❌ 백엔드 엔티티 생성 실패: {e}")
        return None
    
# 백엔드 상태 업데이트 콜백
def notify_backend(task_id: str, status: str, user_id: str):
    try:
        data = {
            "userId": user_id,
            "taskId": task_id,
            "status": status
        }

        # 실제 API 호출 (백엔드가 없으므로 주석 처리)
        # callback_url = f"{settings.API_BASE_URL}/api/status"  # 또는 적절한 경로
        # res = requests.post(settings.CALLBACK_URL, json=data)
        
        print(f"✅ 상태 보고 (시뮬레이션): {status}")
        
    except Exception as e:
        print(f"❌ 상태 보고 실패: {e}")