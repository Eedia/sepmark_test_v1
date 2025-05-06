import os
import subprocess
import glob
from datetime import datetime
import shutil

def extract_frames(video_path, output_dir=None, fps=None):
    """
    비디오를 프레임으로 분리합니다.
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        output_dir (str, optional): 출력 프레임 저장 디렉토리. 기본값은 video_path와 같은 위치에 frames 폴더
        fps (int, optional): 추출할 프레임 레이트. 기본값은 동영상 원본 fps
    
    Returns:
        str: 프레임이 저장된 디렉토리 경로
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        base_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(base_dir, f"{video_name}_frames")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # ffmpeg 명령어 구성
    if fps:
        fps_arg = f"-vf fps={fps}"
    else:
        fps_arg = ""
    
    command = f'ffmpeg -i "{video_path}" {fps_arg} "{output_dir}/%06d.png"'
    
    try:
        # ffmpeg 실행
        subprocess.run(command, shell=True, check=True)
        print(f"✅ 비디오를 프레임으로 성공적으로 분리했습니다.")
        print(f"   프레임 저장 위치: {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 프레임 추출 중 오류 발생: {e}")
        return None

def combine_frames(frames_dir, output_video_path=None, fps=30, codec="libx264", crf=18):
    """
    프레임을 비디오로 합칩니다.
    
    Args:
        frames_dir (str): 프레임 이미지가 있는 디렉토리 경로
        output_video_path (str, optional): 출력 비디오 파일 경로
        fps (int, optional): 비디오 프레임 레이트. 기본값 30
        codec (str, optional): 비디오 코덱. 기본값 libx264
        crf (int, optional): 비디오 품질 (낮을수록 좋음). 기본값 18
    
    Returns:
        str: 생성된 비디오 파일 경로
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"프레임 디렉토리를 찾을 수 없습니다: {frames_dir}")
    
    # 프레임 파일 확인
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not frames:
        raise FileNotFoundError(f"디렉토리에 프레임 이미지가 없습니다: {frames_dir}")
    
    # 출력 비디오 경로 설정
    if output_video_path is None:
        parent_dir = os.path.dirname(frames_dir)
        dir_name = os.path.basename(frames_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(parent_dir, f"{dir_name}_{timestamp}.mp4")
    
    # ffmpeg 명령어 구성
    input_pattern = os.path.join(frames_dir, "%06d.png")
    command = f'ffmpeg -framerate {fps} -i "{input_pattern}" -c:v {codec} -crf {crf} -pix_fmt yuv420p "{output_video_path}"'
    
    try:
        # ffmpeg 실행
        subprocess.run(command, shell=True, check=True)
        print(f"✅ 프레임을 비디오로 성공적으로 합쳤습니다.")
        print(f"   비디오 저장 위치: {output_video_path}")
        return output_video_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 비디오 생성 중 오류 발생: {e}")
        return None

def get_video_info(video_path):
    """
    비디오 파일의 정보를 가져옵니다.
    
    Args:
        video_path (str): 비디오 파일 경로
    
    Returns:
        dict: 비디오 정보 (fps, 해상도, 길이 등)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # FFprobe 명령어로 비디오 정보 가져오기
    command = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of json "{video_path}"'
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        # 결과 파싱 (JSON)
        import json
        info = json.loads(result.stdout)
        stream_info = info.get('streams', [{}])[0]
        
        # 프레임 레이트 계산 (분수 형태로 반환됨)
        fps_fraction = stream_info.get('r_frame_rate', '30/1')
        fps_parts = list(map(float, fps_fraction.split('/')))
        fps = fps_parts[0] / fps_parts[1] if len(fps_parts) > 1 else fps_parts[0]
        
        return {
            'width': int(stream_info.get('width', 0)),
            'height': int(stream_info.get('height', 0)),
            'fps': float(fps),
            'duration': float(stream_info.get('duration', 0)),
            'frames': int(float(fps) * float(stream_info.get('duration', 0)))
        }
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 비디오 정보 가져오기 실패: {e}")
        return None

def clean_frames_directory(frames_dir):
    """
    프레임 디렉토리를 정리합니다 (선택적으로 삭제).
    
    Args:
        frames_dir (str): 프레임 디렉토리 경로
    """
    if os.path.isdir(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            print(f"✅ 프레임 디렉토리 정리 완료: {frames_dir}")
        except Exception as e:
            print(f"⚠️ 프레임 디렉토리 정리 실패: {e}")
    else:
        print(f"⚠️ 프레임 디렉토리를 찾을 수 없습니다: {frames_dir}")
