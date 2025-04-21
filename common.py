import time
import functools
from datetime import datetime

def measure_time(func):
    """
    함수 실행 시간을 측정하는 데코레이터
    
    Args:
        func: 측정할 함수
        
    Returns:
        래핑된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 시작 시간 기록
        start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🕒 [{func.__name__}] 시작: {start_datetime}")
        
        try:
            # 원래 함수 실행
            result = func(*args, **kwargs)
            status = "완료"
        except Exception as e:
            # 에러 발생 시
            result = None
            status = f"실패 - {str(e)}"
            raise
        finally:
            # 종료 시간 기록
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 소요 시간 포맷팅 (시간이 0이면 표시 안함, 분이 0이면 표시 안함)
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            time_parts = []
            if hours > 0:
                time_parts.append(f"{int(hours)}시")
            if hours > 0 or minutes > 0:
                time_parts.append(f"{int(minutes)}분")
            time_parts.append(f"{seconds:.2f}초")
            formatted_time = " ".join(time_parts)
            
            print(f"🏁 [{func.__name__}] {status} - 소요시간: {formatted_time}")
        
        return result
    
    return wrapper