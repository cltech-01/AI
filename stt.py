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




# MPS(Metal Performance Shaders) 활성화 확인 및 설정
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # M시리즈 GPU 사용
    else:
        return torch.device("cpu")

@measure_time
def transcribe_audio_optimized(audio_path, model_name):
    """MPS 백엔드를 활용한 최적화된 Whisper 변환"""
    device = get_device()
    print(f"🖥️ 사용 중인 디바이스: {device}")
    
    try:
        # MPS 백엔드 사용 시도
        model = whisper.load_model(model_name).to(device)
    except:
        # 오류 발생 시 CPU로 폴백
        print(f"⚠️ MPS 디바이스에서 오류 발생, CPU로 대체합니다")
        device = torch.device("cpu")
        model = whisper.load_model(model_name).to(device)
    
    # 모델 추론 최적화 설정
    with torch.inference_mode():
        result = model.transcribe(
            audio_path,
            language="ko",
            fp16=False,
            verbose=True,
            condition_on_previous_text=False
        )
    
    return result

# 큰 오디오 파일을 여러 청크로 나누어 병렬 처리
def process_audio_in_parallel(audio_path, chunk_size, model_name):
    """오디오를 여러 청크로 나누어 병렬 처리"""
    # 1. 오디오를 chunk_size초 단위로 나누기
    # 2. 각 청크를 별도의 프로세스에서 처리
    # 3. 결과 병합
    
    # 멀티프로세싱 풀 생성 (코어 수에 맞게)
    num_processes = multiprocessing.cpu_count() - 1  # 한 코어는 여유롭게
    
    # 이 부분은 실제 구현이 필요함 - 오디오 분할 및 병렬 처리
    
    return "병렬 처리 결과"

# 여러 오디오 파일 동시 처리
def process_multiple_files(audio_files, model_name):
    """여러 오디오 파일을 동시에 처리"""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = pool.starmap(
            transcribe_audio_optimized,
            [(file, model_name) for file in audio_files]
        )
    return results

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
def transcribe_with_segments(audio_path: str, model_name: str) -> dict:
    """
    오디오 파일을 변환하고 세그먼트 정보도 저장합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        model_name: 사용할 Whisper 모델 크기
        
    Returns:
        변환 결과 딕셔너리
    """
    # 오디오 파일 변환
    result = transcribe_audio(audio_path, model_name)
    
    # 세그먼트 정보 저장
    audio_filename = Path(audio_path).stem
    output_dir = Path(audio_path).parent
    segments_path = output_dir / f"{audio_filename}_segments.txt"
    
    with open(segments_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f} → {end:.2f}] {text}\n")
    
    print(f"✅ 세그먼트 정보 저장 완료: {segments_path}")
    print(f"📊 세그먼트 수: {len(result['segments'])} 개")
    
    return result

@measure_time
def process_audio(audio_path, model_name, parallel_threshold_mb=50, chunk_duration=120, username=None):
    """
    오디오 파일을 MPS로 추론하기
    
    Args:
        audio_path: 처리할 오디오 파일 경로
        model_name: Whisper 모델 크기 ("tiny", "base", "small", "medium", "large")
        parallel_threshold_mb: 병렬 처리 시작할 파일 크기 임계값(MB)
        chunk_duration: 병렬 처리시 분할할 청크 길이(초)
        username: 결과를 저장할 사용자 디렉토리 이름
        
    Returns:
        변환 결과 딕셔너리
    """
    # 1. 파일 존재 확인
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
    
    # 2. MPS 디바이스 확인
    device = get_device()
    print(f"🖥️ 사용 중인 디바이스: {device}")
    
    # 3. 오디오 파일이 큰 경우 병렬 처리, 작은 경우 단일 처리
    audio_info = Path(audio_path).stat()
    file_size_mb = audio_info.st_size / (1024 * 1024)
    
    # 파일 크기가 임계값 이상이면 병렬 처리
    if file_size_mb > parallel_threshold_mb:
        print(f"📊 파일 크기: {file_size_mb:.2f}MB - 병렬 처리 시작")
        result = split_and_process_parallel(audio_path, chunk_duration, model_name)
    else:
        print(f"📊 파일 크기: {file_size_mb:.2f}MB - 단일 처리 시작")
        result = transcribe_audio_optimized(audio_path, model_name)
    
    # 4. 결과 저장
    text_path, json_path = save_transcription_results(audio_path, result, username)
    
    return result, text_path

@measure_time
def split_and_process_parallel(audio_path, chunk_duration, model_name):
    """오디오를 청크로 나누어 병렬 처리"""
    # 1. 오디오 로드
    print(f"🔊 오디오 파일 로드 중: {audio_path}")
    audio = AudioSegment.from_wav(audio_path)
    
    # 2. 청크로 분할
    total_duration = len(audio) / 1000  # 초 단위
    num_chunks = int(np.ceil(total_duration / chunk_duration))
    
    print(f"⏱️ 총 오디오 길이: {total_duration:.2f}초, {num_chunks}개 청크로 분할")
    
    # 3. 임시 파일 생성 및 청크 저장
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration * 1000  # ms 단위
        end_time = min((i + 1) * chunk_duration * 1000, len(audio))
        
        chunk = audio[start_time:end_time]
        chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    
    # 4. 병렬 처리
    num_processes = min(multiprocessing.cpu_count() - 1, len(chunk_files))
    print(f"🧵 병렬 처리 시작: {num_processes}개 프로세스로 {len(chunk_files)}개 청크 처리")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_results = pool.starmap(
            process_chunk,
            [(chunk_file, model_name) for chunk_file in chunk_files]
        )
    
    # 5. 결과 합치기
    merged_text = " ".join([result["text"] for result in chunk_results])
    
    # 세그먼트도 합치기 (시간 오프셋 조정 필요)
    merged_segments = []
    time_offset = 0
    
    for i, result in enumerate(chunk_results):
        for segment in result["segments"]:
            segment["start"] += time_offset
            segment["end"] += time_offset
            merged_segments.append(segment)
        
        # 다음 청크의 시간 오프셋 업데이트
        if i < len(chunk_results) - 1:
            time_offset += chunk_duration
    
    # 6. 임시 파일 정리
    for file in chunk_files:
        os.remove(file)
    os.rmdir(temp_dir)
    
    # 7. 최종 결과 반환
    return {
        "text": merged_text,
        "segments": merged_segments
    }

def process_chunk(chunk_path, model_name):
    """개별 청크를 처리 (병렬 프로세스에서 실행)"""
    # 중요: MPS가 아닌 CPU 사용
    device = torch.device("cpu")
    model = whisper.load_model(model_name).to(device)
    
    with torch.inference_mode():
        result = model.transcribe(
            chunk_path,
            language="ko",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False  
        )
    
    return result

def save_transcription_results(audio_path, result, username=None):
    """결과를 파일로 저장"""
    path = Path(audio_path)
    base_dir = path.parent
    stem = path.stem
    
    # 기본 텍스트 저장
    text_path = base_dir / f"{stem}_transcript.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # JSON 파일도 같은 디렉토리에 저장
    json_path = base_dir / f"{stem}.json"
    
    # 세그먼트 데이터 JSON 형식으로 준비
    segments_data = []
    for segment in result["segments"]:
        segments_data.append({
            "start_time": segment["start"],
            "end_time": segment["end"],
            "text": segment["text"]
        })
    
    # JSON 파일 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 텍스트 저장 완료: {text_path}")
    print(f"✅ JSON 세그먼트 저장 완료: {json_path}")
    print(f"🔤 텍스트 길이: {len(result['text'])} 글자")
    print(f"📊 세그먼트 수: {len(result['segments'])} 개")
    return [str(text_path),str(json_path)]

if __name__ == "__main__":
    # 오디오 파일 처리 시작
    process_audio(
        audio_path="./Data/Sound/jhkim/01.wav",
        model_name="small",
        parallel_threshold_mb=50,
        chunk_duration=120,
        username="jhkim"
    )