# frame_extractor.py
"""영상에서 주요 프레임을 추출하는 모듈"""
import cv2
import os
import base64
from pathlib import Path

def extract_frames(video_path, output_dir="frames", interval_seconds=10, max_frames=20):
    """
    영상에서 일정 간격으로 프레임을 추출합니다.
    
    Args:
        video_path: 영상 파일 경로
        output_dir: 프레임 저장 디렉토리
        interval_seconds: 프레임 추출 간격 (초)
        max_frames: 최대 추출 프레임 수
    
    Returns:
        추출된 프레임 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 영상 정보:")
    print(f"  해상도: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"  FPS: {fps:.0f}")
    print(f"  길이: {duration:.0f}초 ({duration/60:.1f}분)")
    
    frame_interval = int(fps * interval_seconds)
    frame_paths = []
    frame_count = 0
    
    while cap.isOpened() and len(frame_paths) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            filename = f"frame_{len(frame_paths):03d}_{int(timestamp)}s.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # 리사이즈 (API 전송 최적화)
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                frame = cv2.resize(frame, (1280, int(height * scale)))
            
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_paths.append(filepath)
            
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            print(f"  ✅ 프레임 {len(frame_paths)}: {minutes}분 {seconds}초")
        
        frame_count += 1
    
    cap.release()
    print(f"\n총 {len(frame_paths)}개 프레임 추출 완료")
    
    return frame_paths

def frame_to_base64(frame_path):
    """프레임 이미지를 Base64로 인코딩"""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_demo_frames():
    """데모용: 영상이 없을 때 텍스트 설명으로 대체"""
    return [
        {"timestamp": "0:30", "description": "워밍업 단계. 가볍게 포핸드를 치고 있음. 스탠스가 좁고 팔만 사용하여 스윙."},
        {"timestamp": "3:00", "description": "포핸드 연습. 라켓을 높이 들어올려 테이크백. 임팩트 시 팔꿈치가 완전히 펴짐."},
        {"timestamp": "7:00", "description": "백핸드 연습 시작. 양손 백핸드, 왼손 그립이 약함. 타점이 몸에 가까움."},
        {"timestamp": "12:00", "description": "서브 연습. 토스가 좌우로 흔들림. 프로네이션 없이 밀어치는 서브."},
        {"timestamp": "17:00", "description": "발리 연습. 라켓을 크게 스윙. 스플릿 스텝 없이 네트 접근."},
        {"timestamp": "22:00", "description": "피로 시작. 실수가 증가. 폼이 흐트러지기 시작."},
        {"timestamp": "27:00", "description": "마지막 연습. 체력 소진. 팔로스루가 짧아지고 공의 깊이가 줄어듦."},
    ]