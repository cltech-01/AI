import requests
import json
import time
import sys
from sseclient import SSEClient

def test_chat_streaming():
    # API 엔드포인트 URL
    url = "http://localhost:8001/chat"
    
    # 테스트 요청 데이터
    data = {
        "userId": "jhkim",  # 테스트할 사용자 ID로 변경
        "message": "이 강의는 어떤 내용을 다루나요?",  # 테스트할 메시지로 변경
        "lectureId": None,  # 선택적 강의 ID
        "conversationId": None  # 새 대화 시작
    }
    
    print("====== 채팅 스트리밍 테스트 시작 ======")
    print(f"사용자: {data['userId']}")
    print(f"질문: {data['message']}")
    print("------------------------------")
    
    # 스트리밍 요청 시작 (POST 요청이므로 헤더에 Accept: text/event-stream을 추가)
    # SSEClient는 일반적으로 GET 요청을 위한 것이므로, 여기서는 requests로 POST 요청을 먼저 보내고
    # 응답의 내용을 한 줄씩 처리하는 방식을 사용
    
    try:
        response = requests.post(
            url, 
            json=data,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        # 응답 상태 확인
        if response.status_code != 200:
            print(f"오류: HTTP {response.status_code}")
            print(response.text)
            return
        
        # 전체 응답 내용
        full_response = ""
        sources = []
        
        # 스트리밍 응답 처리
        for line in response.iter_lines():
            if line:
                # 'data:' 접두사 제거 및 JSON 파싱
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    json_str = line[6:]  # 'data: ' 이후의 문자열
                    try:
                        event_data = json.loads(json_str)
                        
                        # 이벤트 유형에 따른 처리
                        if event_data.get("type") == "start":
                            print(f"대화 ID: {event_data.get('conversation_id')}")
                            print("응답 시작...")
                            
                        elif event_data.get("type") == "chunk":
                            chunk = event_data.get("content", "")
                            print(chunk, end="", flush=True)
                            full_response += chunk
                            
                        elif event_data.get("type") == "sources":
                            sources = event_data.get("sources", [])
                            
                        elif event_data.get("type") == "end":
                            print("\n응답 완료")
                            
                        elif event_data.get("type") == "error":
                            print(f"\n오류: {event_data.get('message')}")
                    
                    except json.JSONDecodeError:
                        print(f"JSON 파싱 오류: {json_str}")
        
        # 참조 문서 출력
        if sources:
            print("\n----- 참조 문서 -----")
            for i, source in enumerate(sources):
                content_preview = source.get("content", "")[:100] + "..." if len(source.get("content", "")) > 100 else source.get("content", "")
                metadata = source.get("metadata", {})
                print(f"[{i+1}] {content_preview}")
                if metadata:
                    print(f"    메타데이터: {metadata}")
                print()
        
        print("====== 테스트 완료 ======")
        
    except requests.RequestException as e:
        print(f"요청 오류: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중단되었습니다.")

if __name__ == "__main__":
    test_chat_streaming() 