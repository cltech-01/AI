import requests
import json
import time
import sys

def test_chat_streaming():
    # API 엔드포인트 URL
    url = "http://localhost:8001/chat"
    
    # 테스트 요청 데이터
    data = {
        "userId": "jhkim",
        "message": "이 강의는 어떤 내용을 다루나요?",
        "lectureId": None,
        "conversationId": None
    }
    
    print("====== 채팅 스트리밍 테스트 시작 ======")
    print(f"사용자: {data['userId']}")
    print(f"질문: {data['message']}")
    print("------------------------------")
    
    try:
        # POST 요청 보내기 - stream=True로 설정하여 스트리밍 응답 받기
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
        
        # 응답 변수 초기화
        full_response = ""
        
        # 타이핑 효과를 위한 변수
        last_update_time = time.time()
        
        # 스트리밍 응답 처리
        buffer = ""
        for chunk in response.iter_content(chunk_size=1):
            if not chunk:
                continue
                
            # 바이트를 문자열로 디코딩
            chunk_str = chunk.decode('utf-8')
            buffer += chunk_str
            
            # 완전한 SSE 이벤트 확인
            if buffer.endswith("\n\n") and buffer.startswith("data: "):
                # data: 접두사 제거 및 JSON 파싱
                try:
                    json_str = buffer.strip()[6:]
                    event_data = json.loads(json_str)
                    
                    # 이벤트 유형에 따른 처리
                    if event_data.get("type") == "start":
                        print(f"대화 ID: {event_data.get('conversation_id')}")
                        print("응답:", end="", flush=True)
                    
                    elif event_data.get("type") == "chunk":
                        # 타이핑 효과를 위한 딜레이
                        current_time = time.time()
                        if current_time - last_update_time > 0.01:
                            sys.stdout.write(event_data.get("content", ""))
                            sys.stdout.flush()
                            last_update_time = current_time
                            full_response += event_data.get("content", "")
                    
                    elif event_data.get("type") == "sources":
                        sources = event_data.get("sources", [])
                        print("\n\n----- 참조 문서 -----")
                        for i, source in enumerate(sources):
                            content = source.get("content", "")[:100]
                            if len(source.get("content", "")) > 100:
                                content += "..."
                            print(f"[{i+1}] {content}")
                            
                            metadata = source.get("metadata", {})
                            if metadata:
                                print(f"    메타데이터: {metadata}")
                    
                    elif event_data.get("type") == "end":
                        print("\n응답 완료")
                        
                    elif event_data.get("type") == "error":
                        print(f"\n오류: {event_data.get('message')}")
                
                except json.JSONDecodeError as e:
                    print(f"\nJSON 파싱 오류: {e} - {buffer}")
                
                # 버퍼 초기화
                buffer = ""
        
        print("\n====== 테스트 완료 ======")
        
    except requests.RequestException as e:
        print(f"요청 오류: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중단되었습니다.")

if __name__ == "__main__":
    test_chat_streaming() 