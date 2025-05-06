import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
import subprocess
from tqdm import tqdm

from Decoder import DW_Tracer, DW_Detector
from Decoder_train import make_fixed_message, calculate_bit_accuracy
from ffmpeg_utils import extract_frames, combine_frames, get_video_info, clean_frames_directory

# 설정 변수들
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MESSAGE_LENGTH = 128
INPUT_RESIZE = 256
SAMPLE_INTERVAL = 1
TRACER_MODEL_PATH = "./saved_models/tracer_model.pth"
DETECTOR_MODEL_PATH = "./saved_models/detector_model.pth"
RESULTS_DIR = "./video_results"
TEMP_FRAMES_DIR = "./temp_frames"
OUTPUT_FRAMES_DIR = "./output_frames"

# 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

def message_to_ascii(message_bits):
    """이진 워터마크를 ASCII 문자로 변환"""
    # 8비트를 1바이트(ASCII 문자)로 그룹화
    message_bytes = []
    for i in range(0, len(message_bits), 8):
        if i + 8 <= len(message_bits):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | int(message_bits[i + j])
            message_bytes.append(byte_val)
    
    # 바이트를 ASCII 문자로 변환
    try:
        ascii_message = ''.join(chr(b) for b in message_bytes if 32 <= b <= 126)  # 출력 가능한 ASCII 문자만
        return ascii_message
    except:
        return "변환 불가능한 메시지"


# 고정 메시지 문자열 및 텐서 설정
FIXED_MESSAGE_STR = "hello world"
fixed_message = make_fixed_message(FIXED_MESSAGE_STR, MESSAGE_LENGTH)

# 이미지 변환 함수
transform = transforms.Compose([
    transforms.Resize((INPUT_RESIZE, INPUT_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 비디오 처리 클래스
class VideoDecoder:
    def __init__(self, tracer_path=TRACER_MODEL_PATH, detector_path=DETECTOR_MODEL_PATH):
        # 모델 로드
        self.tracer = DW_Tracer(MESSAGE_LENGTH).to(DEVICE)
        self.detector = DW_Detector(MESSAGE_LENGTH).to(DEVICE)
        
        if os.path.exists(tracer_path):
            self.tracer.load_state_dict(torch.load(tracer_path, map_location=DEVICE))
            self.tracer.eval()
            print(f"✅ Tracer 모델 로드 완료: {tracer_path}")
        else:
            print(f"⚠️ Tracer 모델을 찾을 수 없습니다: {tracer_path}")
        
        if os.path.exists(detector_path):
            self.detector.load_state_dict(torch.load(detector_path, map_location=DEVICE))
            self.detector.eval()
            print(f"✅ Detector 모델 로드 완료: {detector_path}")
        else:
            print(f"⚠️ Detector 모델을 찾을 수 없습니다: {detector_path}")
    
    def decode_frame(self, frame):
        """단일 프레임에서 워터마크 추출"""
        # PIL 이미지로 변환
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 텐서 변환
        frame_tensor = transform(frame).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # 메시지 추출
            tracer_msg = self.tracer(frame_tensor)
            detector_msg = self.detector(frame_tensor)
            
            # 정확도 계산
            tracer_acc = calculate_bit_accuracy(tracer_msg, fixed_message.to(DEVICE))
            detector_acc = calculate_bit_accuracy(detector_msg, fixed_message.to(DEVICE))
            
            # 메시지 일치 여부 확인
            match_acc = calculate_bit_accuracy(tracer_msg, detector_msg)
            mismatch = 100 - match_acc
            
            # 진위 판단 (mismatch 기준으로)
            is_genuine = mismatch < 40  # 40% 미만의 불일치율은 진짜로 간주
            
            return {
                'tracer_acc': tracer_acc,
                'detector_acc': detector_acc,
                'mismatch': mismatch,
                'is_genuine': is_genuine,
                'tracer_msg': tracer_msg.cpu().numpy(),
                'detector_msg': detector_msg.cpu().numpy()
            }
    
    def create_result_image(self, frame, decode_result, save_path=None):
        """분석 결과를 시각화한 이미지 생성"""
        plt.figure(figsize=(15, 8))  # 더 큰 그림
        
        # 원본 프레임
        plt.subplot(2, 3, 1)
        if isinstance(frame, np.ndarray):
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(frame)
        plt.title("original frame")
        plt.axis('off')
        
        # Tracer 메시지 시각화
        plt.subplot(2, 3, 2)
        tracer_bits = (decode_result['tracer_msg'] > 0).flatten()
        plt.imshow(tracer_bits.reshape(8, 16), cmap='gray')
        plt.title(f"Tracer acc: {decode_result['tracer_acc']:.2f}%")
        plt.axis('off')
        
        # Detector 메시지 시각화
        plt.subplot(2, 3, 3)
        detector_bits = (decode_result['detector_msg'] > 0).flatten()
        plt.imshow(detector_bits.reshape(8, 16), cmap='gray')
        plt.title(f"Detector acc: {decode_result['detector_acc']:.2f}%")
        plt.axis('off')
        
        # ASCII 변환 결과 표시
        tracer_ascii = message_to_ascii(tracer_bits)
        detector_ascii = message_to_ascii(detector_bits)
        
        # Tracer ASCII 표시
        plt.subplot(2, 3, 5)
        plt.text(0.5, 0.5, f"tracer_message: {tracer_ascii}", 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=12)
        plt.title("Tracer extract ASCII")
        plt.axis('off')
        
        # Detector ASCII 표시
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, f"detector message: {detector_ascii}", 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=12)
        plt.title("Detector extract ASCII")
        plt.axis('off')
        
        # 원본 메시지와 비교
        plt.subplot(2, 3, 4)
        original_message = "hello world"  # 원본 메시지
        plt.text(0.5, 0.5, f"orginal_message: {original_message}\n\n"
                        f"Tracer acc: {decode_result['tracer_acc']:.2f}%\n"
                        f"Detector acc: {decode_result['detector_acc']:.2f}%\n"
                        f"Tracer-Detector mismatch: {decode_result['mismatch']:.2f}%", 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=10)
        plt.axis('off')
        
        # 진위 표시
        plt.suptitle(f"mismatch: {decode_result['mismatch']:.2f}% - {'True' if decode_result['is_genuine'] else 'Fake'}", 
                    fontsize=16, 
                    color='green' if decode_result['is_genuine'] else 'red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        
        return plt
    
    def process_video(self, video_path, sample_interval=SAMPLE_INTERVAL, save_results=True, create_output_video=True):
        """비디오 처리 메인 함수"""
        print(f"[처리 시작] 비디오: {video_path}")
        
        # 1. 비디오 정보 가져오기
        video_info = get_video_info(video_path)
        if not video_info:
            print("⚠️ 비디오 정보를 가져올 수 없습니다.")
            return None
        
        print(f"비디오 정보:")
        print(f"  - 해상도: {video_info['width']}x{video_info['height']}")
        print(f"  - FPS: {video_info['fps']}")
        print(f"  - 길이: {video_info['duration']:.2f}초")
        print(f"  - 총 프레임: {video_info['frames']}")
        
        # 2. 임시 디렉토리 생성
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        result_dir = os.path.join(RESULTS_DIR, video_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 3. 비디오에서 프레임 추출 (FFmpeg 사용)
        temp_frames_dir = os.path.join(TEMP_FRAMES_DIR, video_name)
        os.makedirs(temp_frames_dir, exist_ok=True)
        
        # FFmpeg로 프레임 추출
        frames_dir = extract_frames(video_path, output_dir=temp_frames_dir)
        if not frames_dir:
            print("⚠️ 비디오에서 프레임을 추출할 수 없습니다.")
            return None
        
        # 4. 프레임 분석 및 결과 저장
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png') or f.endswith('.jpg')])
        frame_results = []
        
        genuine_count = 0
        forgery_count = 0
        
        # 결과 로그 파일
        log_file = os.path.join(result_dir, "detection_results.csv")
        with open(log_file, 'w') as f:
            f.write("frame,tracer_acc,detector_acc,mismatch,is_genuine\n")
        
        # 진행바 생성
        total_frames = len(frame_files)
        pbar = tqdm(total=total_frames//sample_interval, desc="프레임 분석 중")
        
        # 출력 프레임 디렉토리
        output_frames_dir = os.path.join(OUTPUT_FRAMES_DIR, video_name)
        os.makedirs(output_frames_dir, exist_ok=True)
        
        for i, frame_file in enumerate(frame_files):
            # 샘플링 간격에 따라 프레임 처리
            if i % sample_interval != 0:
                continue
                
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            # 프레임 디코딩
            result = self.decode_frame(frame)
            frame_results.append(result)
            
            # 진위 여부 카운팅
            if result['is_genuine']:
                genuine_count += 1
            else:
                forgery_count += 1
            
            # CSV 로그 저장
            with open(log_file, 'a') as f:
                f.write(f"{frame_file},{result['tracer_acc']:.2f},{result['detector_acc']:.2f},{result['mismatch']:.2f},{result['is_genuine']}\n")
            
            # 결과 이미지 저장
            if save_results:
                result_image_path = os.path.join(result_dir, f"result_{frame_file}")
                self.create_result_image(frame, result, save_path=result_image_path)
                
                # 진위 판단 표시한 출력 프레임 생성 (비디오 만들기용)
                if create_output_video:
                    # 원본 프레임 복사
                    output_frame = frame.copy()
                    
                    # 진위 여부에 따른 테두리 색상 설정
                    color = (0, 255, 0) if result['is_genuine'] else (0, 0, 255)  # Green for genuine, Red for fake
                    
                    # 프레임에 테두리 그리기
                    thickness = 10
                    cv2.rectangle(output_frame, (0, 0), (output_frame.shape[1], output_frame.shape[0]), color, thickness)
                    
                    # 텍스트 정보 추가
                    text = f"{'Genuine' if result['is_genuine'] else 'Fake'} (Mismatch: {result['mismatch']:.1f}%)"
                    cv2.putText(output_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    
                    # 출력 프레임 저장
                    output_frame_path = os.path.join(output_frames_dir, frame_file)
                    cv2.imwrite(output_frame_path, output_frame)
            
            pbar.update(1)
        
        pbar.close()
        
        # 5. 결과 요약
        total_analyzed = genuine_count + forgery_count
        genuine_percent = (genuine_count / total_analyzed * 100) if total_analyzed > 0 else 0
        forgery_percent = (forgery_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        print("\n=== 비디오 분석 결과 ===")
        print(f"총 분석한 프레임: {total_analyzed}")
        print(f"진짜로 판별된 프레임: {genuine_count} ({genuine_percent:.1f}%)")
        print(f"가짜로 판별된 프레임: {forgery_count} ({forgery_percent:.1f}%)")
        
        # 출력 결과 그래프 생성
        plt.figure(figsize=(12, 6))
        
        # 프레임별 불일치율 그래프
        plt.subplot(1, 2, 1)
        frame_indices = list(range(0, len(frame_files), sample_interval))[:len(frame_results)]
        mismatches = [result['mismatch'] for result in frame_results]
        plt.plot(frame_indices, mismatches, 'b-')
        plt.axhline(y=40, color='r', linestyle='--')  # 기준선 (40%)
        plt.xlabel('frame index')
        plt.ylabel('mismatch (%)')
        plt.title('frame Tracer-Detector mismatch')
        
        # 진위 판별 비율 파이 차트
        plt.subplot(1, 2, 2)
        plt.pie([genuine_count, forgery_count], 
                labels=['True', 'Fake'], 
                autopct='%1.1f%%',
                colors=['green', 'red'])
        plt.title('Record the authenticity of the video')
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "summary.png"))
        plt.close()
        
        # 6. 출력 비디오 생성 (선택적)
        output_video_path = None
        if create_output_video and os.listdir(output_frames_dir):
            output_video_path = os.path.join(result_dir, f"{video_name}_analyzed.mp4")
            combine_frames(output_frames_dir, output_video_path, fps=video_info['fps'])
            print(f"✅ 분석 결과 비디오 생성 완료: {output_video_path}")
        
        # 7. 임시 파일 정리 (선택적)
        if os.environ.get('KEEP_TEMP_FILES', '0') != '1':
            clean_frames_directory(temp_frames_dir)
        
        return {
            'video_path': video_path,
            'total_frames': total_analyzed,
            'genuine_frames': genuine_count,
            'forgery_frames': forgery_count,
            'genuine_percent': genuine_percent,
            'output_video_path': output_video_path,
            'result_dir': result_dir
        }


# 실행 함수
def decode_video(video_path, sample_interval=SAMPLE_INTERVAL):
    """비디오 디코딩 인터페이스 함수"""
    decoder = VideoDecoder()
    return decoder.process_video(video_path, sample_interval=sample_interval)


# 메인 실행 부분
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='SepMark 비디오 워터마크 디코더')
    parser.add_argument('video_path', help='분석할 비디오 파일 경로')
    parser.add_argument('--interval', type=int, default=SAMPLE_INTERVAL, help='프레임 샘플링 간격 (기본값: 10)')
    parser.add_argument('--no-video', action='store_false', dest='create_video', help='분석 결과 비디오를 생성하지 않음')
    
    args = parser.parse_args()
    
    if os.path.exists(args.video_path):
        decode_video(args.video_path, args.interval)
    else:
        print(f"⚠️ 입력 비디오를 찾을 수 없습니다: {args.video_path}")
