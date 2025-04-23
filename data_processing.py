import os
import re
import json
import subprocess
from dotenv import load_dotenv
from common import measure_time

load_dotenv()


@measure_time
def extract_audio_from_video(video_path: str, username: str, video_uuid: str) -> str:
    """
    영상 파일에서 오디오를 추출하고 전처리합니다.
    
    Args:
        video_path: 입력 영상 파일 경로
        username: 사용자 이름
        video_uuid: 비디오 UUID
        
    Returns:
        출력 오디오 파일 경로
    """
    # 출력 디렉토리 생성
    output_dir = f"./Data/Sound/{username}"
    os.makedirs(output_dir, exist_ok=True)

    # 출력 파일 경로
    audio_output_path = f"{output_dir}/{video_uuid}.wav"

    # ffmpeg 명령어 구성 (로그 레벨 quiet 추가)
    cmd = [
        "ffmpeg", "-y", 
        "-loglevel", "error",  # 에러만 표시
        "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1",
        "-af", "silenceremove=start_periods=1:start_duration=1:start_threshold=-45dB:stop_periods=-1:stop_duration=1:stop_threshold=-45dB,aresample=async=1",
        "-c:a", "pcm_s16le", 
        audio_output_path
    ]
    
    # subprocess로 실행하고 출력 제어
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✅ 오디오 추출 및 전처리 완료: {audio_output_path}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 실행 오류: {e.stderr.decode() if e.stderr else '알 수 없는 오류'}")
        raise Exception(f"오디오 추출 실패")


# 텍스트에서 불필요한 문자, 반복적인 의미없는 어구를 제외합니다. 
def clean_text(file_path: str, use_nlp: bool = False) -> str:
    # 기본 정제 - 한국어, 영어, 숫자, 기본 문장부호만 유지
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r'[^\w\s\.,\?\!\'\":;-]', ' ', text)
    
    # 반복적인 의미없는 어구 제거 (음..., 어..., 그..., 그래서... 등)
    filler_words = [
        r'음+\.{0,3}\s*', r'어+\.{0,3}\s*', r'그+\.{0,3}\s*', 
        r'아+\.{0,3}\s*', r'네+\.{0,3}\s*', r'뭐+\.{0,3}\s*',
        r'그니까요\.{0,3}\s*', r'그래요\.{0,3}\s*', r'그렇죠\.{0,3}\s*',
        r'이제\s+', r'그러면\s+', r'그러니까\s+', r'그래서\s+'
    ]
    
    for pattern in filler_words:
        # 연속으로 반복되는 경우 한 번만 남김
        text = re.sub(f'({pattern})\\1+', '\\1', text)
        
    # 반복되는 문장 패턴 제거 (완전히 동일한 문장 반복)
    lines = text.split('\n')
    unique_lines = []
    for line in lines:
        if line and line not in unique_lines:
            unique_lines.append(line)
    text = '\n'.join(unique_lines)
    
    # 여러 개의 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    
    # 문장 내 3번 이상 반복되는 단어 패턴 제거
    words = text.split()
    cleaned_words = []
    i = 0
    while i < len(words):
        if i < len(words) - 2 and words[i] == words[i+1] == words[i+2]:
            # 반복된 단어는 한 번만 추가
            cleaned_words.append(words[i])
            # 같은 단어가 더 이상 연속되지 않을 때까지 건너뛰기
            while i < len(words) - 1 and words[i] == words[i+1]:
                i += 1
        else:
            cleaned_words.append(words[i])
        i += 1
    
    text = ' '.join(cleaned_words).strip()
    
    
    dir_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    new_file_name = f"cleaned_{file_name}"
    output_path = f"{dir_path}/{new_file_name}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ 텍스트 정제 완료: {output_path}")
    
    return output_path

@measure_time
def compress_video_to_h264(video_path: str) -> str:
    """
    입력 영상 파일을 H.264 코덱으로 압축한 후 원본 파일을 대체합니다.

    Args:
        video_path (str): 원본 영상 경로

    Returns:
        str: 대체된 영상 파일 경로
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {video_path}")
    
    dir_name = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_output_path = os.path.join(dir_name, f"{base_name}_h264_temp.mp4")

    # ffmpeg 명령어 (H.264로 압축)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",            # 🔄 H.264로 변경
        "-preset", "fast",
        "-crf", "23",                 # H.264용 추천 값 (품질/용량 밸런스)
        "-c:a", "aac",
        temp_output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # 기존 파일 삭제 후 대체
        os.remove(video_path)
        os.rename(temp_output_path, video_path)
        print(f"✅ H.264 압축 완료 및 원본 대체: {video_path}")
        return video_path
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 압축 오류: {e.stderr.decode() if e.stderr else '알 수 없는 오류'}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        raise Exception("H.264 압축 실패")



if __name__ == "__main__":
    clean_text("./reference/cleaned_example.txt", use_nlp=True)
