import os
import subprocess
import threading
import json

# ✅ FPS 추출 함수
def get_video_fps(video_path):
    try:
        command = [
            "ffprobe.exe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        rate_str = data["streams"][0]["r_frame_rate"]
        num, den = map(int, rate_str.split('/'))
        fps = num / den
        return round(fps, 2)
    except Exception as e:
        print("❌ FPS 추출 실패:", e)
        return 30  # fallback

# ✅ 영상 길이 추출
def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe.exe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print("❌ 영상 길이 추출 실패:", e)
        return None

# ✅ 프레임 + 오디오 추출 (전체 프레임 추출)
def extract_frames(video_path):
    print(" 영상 경로:", video_path)
    if not os.path.exists(video_path):
        print("❌ 영상 파일이 존재하지 않습니다1111.")
        return

    video_title = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), video_title)
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%04d.png")
    audio_output_path = os.path.join(output_dir, "audio.aac")

    # ▶ 프레임 전체 추출
    def extract_video_frames():
        command = [
            "ffmpeg.exe",
            "-i", video_path,
            output_pattern
        ]
        print("🎥 실행 커맨드(프레임):", " ".join(command))
        try:
            subprocess.run(command, check=True)
            print(f"✅ 프레임 추출 완료: {output_dir}")
        except subprocess.CalledProcessError:
            print("❌ ffmpeg 실행 실패 (프레임)")

    # ▶ 오디오 추출
    def extract_audio():
        command = [
            "ffmpeg.exe",
            "-i", video_path,
            "-vn",
            "-acodec", "copy",
            audio_output_path
        ]
        print("🎧 실행 커맨드(오디오):", " ".join(command))
        try:
            subprocess.run(command, check=True)
            print(f"✅ 오디오 추출 완료: {audio_output_path}")
        except subprocess.CalledProcessError:
            print("❌ ffmpeg 실행 실패 (오디오)")

    # 병렬 실행
    t1 = threading.Thread(target=extract_video_frames)
    t2 = threading.Thread(target=extract_audio)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# ✅ 프레임 + 오디오 병합 (원본 FPS 고정)
def combine_frames(output_dir, video_path, output_name="output_encoded.mp4"):
    frame_pattern = os.path.join(output_dir, "encoded_frame_%04d.png")
    first_frame = frame_pattern.replace("%04d", "0001")

    # 오디오 파일 확인
    audio_file = None
    for ext in ["audio.aac", "audio.mp3"]:
        candidate = os.path.join(output_dir, ext)
        if os.path.exists(candidate):
            audio_file = candidate
            break

    if not os.path.exists(first_frame):
        print("❌ 프레임 이미지가 존재하지 않습니다.")
        return

    if not os.path.exists(video_path):
        print("❌ 원본 영상 경로가 잘못되었습니다.")
        return

    # ✅ 원본 FPS 가져오기
    fps = get_video_fps(video_path)
    print(f"🎯 원본 FPS 사용: {fps}")

    output_path = os.path.join(output_dir, output_name)

    # 병합 명령 구성 (입력/출력 둘 다 FPS 고정)
    if audio_file:
        command = [
            "ffmpeg.exe",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", audio_file,
            "-c:v", "libx264",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path
        ]
    else:
        command = [
            "ffmpeg.exe",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            output_path
        ]

    print("🎬 병합 커맨드:", " ".join(command))
    try:
        subprocess.run(command, check=True)
        print(f"✅ 병합 완료! 결과 파일: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ 병합 실패:", e)
