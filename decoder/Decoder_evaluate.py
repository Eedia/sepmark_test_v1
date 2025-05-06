import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from Decoder import DW_Tracer, DW_Detector
from Noise_Layer import RandomForwardNoisePool
from Decoder_train import make_fixed_message, calculate_bit_accuracy

# 설정 변수들
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MESSAGE_LENGTH = 128
INPUT_RESIZE = 256
TRACER_MODEL_PATH = "./saved_models/tracer_model.pth"
DETECTOR_MODEL_PATH = "./saved_models/detector_model.pth"
RESULTS_DIR = "./eval_results"

# 고정 메시지 문자열 및 텐서 설정
FIXED_MESSAGE_STR = "hello world"
fixed_message = make_fixed_message(FIXED_MESSAGE_STR, MESSAGE_LENGTH)

# 경로 및 디렉터리 설정
os.makedirs(RESULTS_DIR, exist_ok=True)

# 이미지 변환 함수
transform = transforms.Compose([
    transforms.Resize((INPUT_RESIZE, INPUT_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 평가 함수
def evaluate_decoders(test_img_path):
    # 두 디코더 모델 불러오기
    tracer = DW_Tracer(MESSAGE_LENGTH).to(DEVICE)
    detector = DW_Detector(MESSAGE_LENGTH).to(DEVICE)
    
    # 저장된 모델 로드
    tracer.load_state_dict(torch.load(TRACER_MODEL_PATH, map_location=DEVICE))
    detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
    
    tracer.eval()
    detector.eval()
    
    # 테스트 이미지 변환
    test_img = Image.open(test_img_path).convert('RGB')
    test_tensor = transform(test_img).unsqueeze(0).to(DEVICE)
    
    # 노이즈 레이어 초기화
    noise_pool = RandomForwardNoisePool()
    
    # 다양한 왜곡 테스트
    distortion_types = {
        '원본': lambda x: x,
        'JPEG 압축': lambda x: noise_pool.jpeg_compression(x),
        '리사이즈': lambda x: noise_pool.resize(x),
        '가우시안 블러': lambda x: noise_pool.gaussian_blur(x),
        '밝기 조정': lambda x: noise_pool.brightness_adjust(x),
        '노이즈 추가': lambda x: noise_pool.gaussian_noise(x),
        'SimSwap 시뮬레이션': lambda x: noise_pool.simswap_simulation(x),
        'GANimation 시뮬레이션': lambda x: noise_pool.ganimation_simulation(x),
        'StarGAN 시뮬레이션': lambda x: noise_pool.stargan_simulation(x)
    }
    
    results = {}
    
    with torch.no_grad():
        for name, distortion_fn in distortion_types.items():
            distorted = distortion_fn(test_tensor)
            
            # Tracer와 Detector에서 메시지 추출
            tracer_msg = tracer(distorted)
            detector_msg = detector(distorted)
            
            # 정확도 계산
            tracer_acc = calculate_bit_accuracy(tracer_msg, fixed_message.to(DEVICE))
            detector_acc = calculate_bit_accuracy(detector_msg, fixed_message.to(DEVICE))
            
            # 일치 여부 확인 (Tracer와 Detector 메시지 비교)
            match_acc = calculate_bit_accuracy(tracer_msg, detector_msg)
            mismatch = 100 - match_acc
            
            results[name] = {
                'tracer_acc': tracer_acc,
                'detector_acc': detector_acc,
                'mismatch': mismatch
            }
            
            print(f"\n{name} 왜곡 결과:")
            print(f"  Tracer 정확도: {tracer_acc:.2f}%")
            print(f"  Detector 정확도: {detector_acc:.2f}%")
            print(f"  불일치율: {mismatch:.2f}%")
            
            # 이미지 시각화 저장
            plt.figure(figsize=(15, 5))
            
            # 왜곡된 이미지
            plt.subplot(1, 3, 1)
            img = distorted[0].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2.0  # -1~1 범위를 0~1 범위로 변환
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(f"{name} 왜곡 이미지")
            plt.axis('off')
            
            # Tracer 메시지 시각화
            plt.subplot(1, 3, 2)
            tracer_bits = (tracer_msg > 0).float().cpu().numpy().flatten()
            plt.imshow(tracer_bits.reshape(8, 16), cmap='gray')
            plt.title(f"Tracer 추출 메시지 (정확도: {tracer_acc:.2f}%)")
            plt.axis('off')
            
            # Detector 메시지 시각화
            plt.subplot(1, 3, 3)
            detector_bits = (detector_msg > 0).float().cpu().numpy().flatten()
            plt.imshow(detector_bits.reshape(8, 16), cmap='gray')
            plt.title(f"Detector 추출 메시지 (정확도: {detector_acc:.2f}%)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/{name}_test.png")
            plt.close()
    
    # 통합 결과 그래프
    plt.figure(figsize=(15, 10))
    distortion_names = list(results.keys())
    
    # Tracer 정확도 그래프
    plt.subplot(3, 1, 1)
    plt.bar(distortion_names, [results[name]['tracer_acc'] for name in distortion_names])
    plt.ylabel('정확도 (%)')
    plt.title('Tracer 정확도')
    plt.xticks(rotation=45)
    plt.ylim(0, 105)
    
    # Detector 정확도 그래프
    plt.subplot(3, 1, 2)
    plt.bar(distortion_names, [results[name]['detector_acc'] for name in distortion_names])
    plt.ylabel('정확도 (%)')
    plt.title('Detector 정확도')
    plt.xticks(rotation=45)
    plt.ylim(0, 105)
    
    # 불일치율 그래프
    plt.subplot(3, 1, 3)
    plt.bar(distortion_names, [results[name]['mismatch'] for name in distortion_names])
    plt.ylabel('불일치율 (%)')
    plt.title('Tracer-Detector 불일치율')
    plt.xticks(rotation=45)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/overall_results.png")
    plt.close()
    
    print("\n✅ 모든 왜곡 테스트 완료")
    
    # 결과에 따른 SepMark 성능 판단
    common_distortions = ['원본', 'JPEG 압축', '리사이즈', '가우시안 블러', '밝기 조정', '노이즈 추가']
    malicious_distortions = ['SimSwap 시뮬레이션', 'GANimation 시뮬레이션', 'StarGAN 시뮬레이션']
    
    # 일반 왜곡에 대한 평균 성능
    avg_tracer_common = sum([results[name]['tracer_acc'] for name in common_distortions]) / len(common_distortions)
    avg_detector_common = sum([results[name]['detector_acc'] for name in common_distortions]) / len(common_distortions)
    avg_mismatch_common = sum([results[name]['mismatch'] for name in common_distortions]) / len(common_distortions)
    
    # 악의적 왜곡에 대한 평균 성능
    avg_tracer_malicious = sum([results[name]['tracer_acc'] for name in malicious_distortions]) / len(malicious_distortions)
    avg_detector_malicious = sum([results[name]['detector_acc'] for name in malicious_distortions]) / len(malicious_distortions)
    avg_mismatch_malicious = sum([results[name]['mismatch'] for name in malicious_distortions]) / len(malicious_distortions)
    
    print("\n=== SepMark 성능 분석 ===")
    print("\n[일반 왜곡에 대한 평균 성능]")
    print(f"Tracer 정확도: {avg_tracer_common:.2f}%")
    print(f"Detector 정확도: {avg_detector_common:.2f}%")
    print(f"불일치율: {avg_mismatch_common:.2f}%")
    
    print("\n[악의적 왜곡에 대한 평균 성능]")
    print(f"Tracer 정확도: {avg_tracer_malicious:.2f}%")
    print(f"Detector 정확도: {avg_detector_malicious:.2f}%")
    print(f"불일치율: {avg_mismatch_malicious:.2f}%")
    
    # 성능 기준 평가
    success = True
    failure_reasons = []
    
    # 이상적인 SepMark 성능 기준
    if avg_tracer_common < 85:
        success = False
        failure_reasons.append(f"Tracer의 일반 왜곡 정확도가 너무 낮습니다: {avg_tracer_common:.2f}%")
    
    if avg_detector_common < 85:
        success = False
        failure_reasons.append(f"Detector의 일반 왜곡 정확도가 너무 낮습니다: {avg_detector_common:.2f}%")
    
    if avg_tracer_malicious < 75:
        success = False
        failure_reasons.append(f"Tracer의 악의적 왜곡 정확도가 너무 낮습니다: {avg_tracer_malicious:.2f}%")
    
    if avg_detector_malicious > 60:
        success = False
        failure_reasons.append(f"Detector의 악의적 왜곡 정확도가 너무 높습니다: {avg_detector_malicious:.2f}%")
    
    if avg_mismatch_common > 20:
        success = False
        failure_reasons.append(f"일반 왜곡에서 불일치율이 너무 높습니다: {avg_mismatch_common:.2f}%")
    
    if avg_mismatch_malicious < 40:
        success = False
        failure_reasons.append(f"악의적 왜곡에서 불일치율이 너무 낮습니다: {avg_mismatch_malicious:.2f}%")
    
    if success:
        print("\n✅ SepMark 디코더가 성공적으로 학습되었습니다!")
    else:
        print("\n❌ SepMark 디코더에 개선이 필요합니다:")
        for reason in failure_reasons:
            print(f"- {reason}")


# 메인 실행 부분
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
        
        if os.path.exists(test_img_path):
            print(f"[테스트 시작] 이미지: {test_img_path}")
            evaluate_decoders(test_img_path)
        else:
            print(f"❌ 테스트 이미지를 찾을 수 없습니다: {test_img_path}")
    else:
        print("❌ 테스트 이미지 경로를 지정해야 합니다.")
        print("사용법: python Decoder_evaluate.py <이미지_경로>")
