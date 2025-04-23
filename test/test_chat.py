import requests
import json
import time
import sys

def test_chat_streaming():
    # API 엔드포인트 URL
    url = "http://localhost:8000/chat"
    
    # 테스트 요청 데이터
    data = {
        "userId": "jhkim",
        "message": "DevOps팀에서 사용하는 도구에 대해서 알려줘",
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
        
        # 전체 응답 저장 변수
        full_response = ""
        
        # 버퍼 준비
        buffer = ""
        
        # 스트리밍 응답 처리 - 라인 단위로 처리
        for line in response.iter_lines():
            if not line:
                continue
                
            # 바이트를 문자열로 디코딩 (한 줄씩 읽어 디코딩 문제 해결)
            line_str = line.decode('utf-8')
            
            # data: 접두사로 시작하는 SSE 이벤트 라인 확인
            if line_str.startswith("data: "):
                # data: 접두사 제거 및 JSON 파싱
                try:
                    json_str = line_str[6:]  # 'data: ' 이후의 문자열
                    event_data = json.loads(json_str)
                    
                    # 이벤트 유형에 따른 처리
                    if event_data.get("type") == "start":
                        print(f"대화 ID: {event_data.get('conversation_id')}")
                        print("응답: ", end="", flush=True)
                    
                    elif event_data.get("type") == "chunk":
                        chunk = event_data.get("content", "")
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        full_response += chunk
                    
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
                    print(f"\nJSON 파싱 오류: {e} - {line_str}")
        
        print("\n====== 테스트 완료 ======")
        
    except requests.RequestException as e:
        print(f"요청 오류: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중단되었습니다.")

if __name__ == "__main__":
    test_chat_streaming() 