import os
import json
import torch
import pydub
import whisper
import tempfile
import multiprocessing
import numpy as np
from pathlib import Path
from config import settings 
from common import measure_time
from pydub import AudioSegment

@measure_time
def transcribe_audio(audio_path: str, model_name: str) -> dict:
    """
    Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        model_name: 사용할 Whisper 모델 크기 ("tiny", "base", "small", "medium", "large")
        
    Returns:
        변환된 텍스트와 세그먼트 정보를 포함한 딕셔너리
    """
    print(f"🔊 Whisper {model_name} 모델로 음성 인식 시작...")
    
    # Whisper 모델 로드
    model = whisper.load_model(model_name).to(get_device())
    
    # 오디오 파일 변환
    result = model.transcribe(
        audio_path,
        language="ko",  # 한국어로 설정
        fp16=False,     # GPU가 없는 경우 False로 설정
        verbose=True,    # 진행 상황 표시
        condition_on_previous_text=False
    )
    
    # 결과 저장 경로
    audio_filename = Path(audio_path).stem
    output_dir = Path(audio_path).parent
    text_path = output_dir / f"{audio_filename}.txt"
    
    # 전체 텍스트 저장
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"✅ 텍스트 저장 완료: {text_path}")
    
    return result


@measure_time
def process_audio(audio_path, model_name, username):
    """
    오디오 파일을 CPU로 추론하기 (멀티프로세싱 없이)
    """
    # 1. 파일 존재 확인
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
    
    # 2. 무조건 CPU 디바이스 사용
    device = torch.device("cpu")
    print(f"🖥️ 사용 중인 디바이스: {device}")
    
    # 3. 단일 프로세스로 처리
    print(f"📊 단일 프로세스로 처리 시작")
    
    # 모델 로드 (CPU에서만)
    model = whisper.load_model(model_name).to(device)
    
    # 추론 실행
    with torch.inference_mode():
        result = model.transcribe(
            audio_path,
            language="ko",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False
        )
    
    # 4. 결과 저장
    text_path, json_path = save_transcription_results(audio_path, result, username)
    
    return result, text_path


def save_transcription_results(audio_path, result, username=None):
    """결과를 Text/{username}에 저장"""
    stem = Path(audio_path).stem
    base_dir = Path(f"./Data/Text/{username}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 텍스트 저장
    text_path = base_dir / f"{stem}_transcript.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # 세그먼트 저장
    json_path = base_dir / f"{stem}.json"
    segments_data = [
        {"start_time": s["start"], "end_time": s["end"], "text": s["text"]}
        for s in result["segments"]
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 텍스트 저장 완료: {text_path}")
    print(f"✅ JSON 세그먼트 저장 완료: {json_path}")
    return [str(text_path), str(json_path)]

if __name__ == "__main__":
    # 오디오 파일 처리 시작
    process_audio(
        audio_path="./reference/example.wav",
        model_name="small",
        parallel_threshold_mb=50,
        chunk_duration=120,
        username="jhkim"
    )