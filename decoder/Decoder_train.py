import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Decoder import DW_Tracer, DW_Detector
from Noise_Layer import RandomForwardNoisePool

# 설정 변수들
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MESSAGE_LENGTH = 128
INPUT_RESIZE = 256
BATCH_SIZE = 32   #32
LEARNING_RATE = 1e-4
EPOCHS = 5
SAVE_INTERVAL = 10
VAL_SPLIT = 0.8
TRACER_MODEL_PATH = "./saved_models/tracer_model.pth"
DETECTOR_MODEL_PATH = "./saved_models/detector_model.pth"
RESULTS_DIR = "./results"
MAX_SAMPLES = 1000   # 최대 100개 이미지만 사용



# 경로 및 디렉터리 설정
ENCODED_DIR = "../test_output_encoded_val_128"  # 워터마크가 삽입된 이미지 디렉토리
ORIGINAL_DIR = "../dataset/val_128"              # 원본 이미지 디렉토리
os.makedirs("./saved_models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 메시지 생성 함수 (Encoder_main.py와 동일하게 메시지 생성)
def make_fixed_message(text, length):
    text_bytes = text.encode('utf-8')
    text_tensor = torch.tensor(list(text_bytes), dtype=torch.float32)

    if text_tensor.numel() >= length:
        message = text_tensor[:length]
    else:
        pad_size = length - text_tensor.numel()
        message = torch.cat([text_tensor, torch.zeros(pad_size)], dim=0)

    # -0.1~0.1 범위로 정규화
    message = (message - 128) / 128 * 0.2
    return message

# 고정 메시지 문자열 및 텐서 설정
FIXED_MESSAGE_STR = "hello world"
fixed_message = make_fixed_message(FIXED_MESSAGE_STR, MESSAGE_LENGTH)

# 정확도 계산 함수
def calculate_bit_accuracy(pred, target, threshold=0):
    # 이진 메시지로 변환 (-0.1~0.1 범위 → 0 또는 1)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 정확도 계산 (정확히 맞춘 비트 / 전체 비트)
    accuracy = (pred_binary == target_binary).float().mean().item()
    return accuracy * 100  # 퍼센트로 변환

# 워터마크된 이미지 데이터셋
class WatermarkedImageDataset(Dataset):
    def __init__(self, encoded_dir, original_dir, message, transform=None):
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"타겟 디렉토리: {os.path.abspath(ENCODED_DIR)}")

        # 인코딩된 이미지 경로 로드
        encoded_files = [f for f in os.listdir(encoded_dir) if f.endswith('.png')]
        self.encoded_paths = sorted([os.path.join(encoded_dir, f) for f in encoded_files])
        print(f"인코딩된 이미지 개수: {len(self.encoded_paths)}")
        
        # 원본 이미지 경로 로드 - 인코딩된 이미지 이름에서 숫자 부분 추출하여 매핑
        self.original_paths = []
        for encoded_file in encoded_files:
            # 'encoded_00001.png'에서 '00001' 추출
            if 'encoded_' in encoded_file:
                number_part = encoded_file.replace('encoded_', '').split('.')[0]
                original_file = f"{number_part}.png"  # '00001.png' 형식
            else:
                # 다른 패턴의 경우 처리
                original_file = encoded_file
                
            original_path = os.path.join(original_dir, original_file)
            if os.path.exists(original_path):
                self.original_paths.append(original_path)
            else:
                print(f"⚠️ 원본 파일을 찾을 수 없음: {original_file}")
        
        # 매칭된 파일만 유지
        if len(self.original_paths) < len(self.encoded_paths):
            print(f"⚠️ 원본 파일을 찾지 못한 인코딩 이미지가 있습니다. 매칭된 이미지만 사용합니다.")
            # 매칭된 인코딩 이미지 경로만 유지
            matched_indices = [i for i, path in enumerate(self.encoded_paths) 
                            if i < len(self.original_paths)]
            self.encoded_paths = [self.encoded_paths[i] for i in matched_indices]
        
        print(f"매칭된 이미지 쌍 개수: {len(self.encoded_paths)}")
        
        # 데이터셋 크기 제한
        if MAX_SAMPLES > 0 and MAX_SAMPLES < len(self.encoded_paths):
            print(f"⚠️ 데이터셋을 {MAX_SAMPLES}개로 제한합니다")
            self.encoded_paths = self.encoded_paths[:MAX_SAMPLES]
            self.original_paths = self.original_paths[:MAX_SAMPLES]
        
        print(f"사용할 이미지 개수: {len(self.encoded_paths)}")
        self.message = message
        self.transform = transform or transforms.Compose([
            transforms.Resize((INPUT_RESIZE, INPUT_RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.encoded_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self.encoded_paths):
            idx = idx % len(self.encoded_paths)
            
        try:
            encoded_img = Image.open(self.encoded_paths[idx]).convert('RGB')
            original_img = Image.open(self.original_paths[idx]).convert('RGB')
            
            if self.transform:
                encoded_img = self.transform(encoded_img)
                original_img = self.transform(original_img)
                
            return encoded_img, original_img, self.message
        except Exception as e:
            print(f"⚠️ 이미지 로딩 오류 (idx={idx}): {e}")
            # 오류 발생 시 첫 번째 이미지 반환 (안전 조치)
            return self.__getitem__(0)


# 학습 함수
def train_decoder(model_type='tracer'):
    print(f"\n{'='*50}")
    print(f"[{model_type.upper()} 모델 학습 시작]")
    print(f"{'='*50}")
    
    # 데이터셋 및 데이터로더 설정
    print("1. 데이터셋 로딩 중...")
    dataset = WatermarkedImageDataset(ENCODED_DIR, ORIGINAL_DIR, fixed_message)
    
    train_size = int(VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    print(f"2. 데이터셋 분할: 학습={train_size}개, 검증={val_size}개")
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"3. 데이터로더 초기화: 배치 크기={BATCH_SIZE}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"4. 총 배치 수: 학습={len(train_loader)}개, 검증={len(val_loader)}개")
    
    # 모델 초기화
    print(f"5. {model_type.capitalize()} 모델 초기화 중...")
    if model_type == 'tracer':
        model = DW_Tracer(MESSAGE_LENGTH).to(DEVICE)
        model_path = TRACER_MODEL_PATH
    else:  # detector
        model = DW_Detector(MESSAGE_LENGTH).to(DEVICE)
        model_path = DETECTOR_MODEL_PATH
    
    # 노이즈 레이어, 손실 함수, 옵티마이저 초기화
    print("6. 노이즈 레이어 및 옵티마이저 초기화 중...")
    noise_pool = RandomForwardNoisePool()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # 학습 결과 추적
    print("7. 학습 변수 초기화 중...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    print(f"\n8. 학습 시작 ({EPOCHS} 에포크)...\n")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        start_time = time.time()
        
        print(f"🔄 에포크 {epoch+1}/{EPOCHS} 시작")
        
        for batch_idx, (encoded_imgs, original_imgs, target_msgs) in enumerate(train_loader):
            # 진행상황 출력
            if batch_idx % 2 == 0 or batch_idx == len(train_loader) - 1:
                print(f"   배치 진행: {batch_idx+1}/{len(train_loader)} ({(batch_idx+1)/len(train_loader)*100:.1f}%)")
            
            encoded_imgs = encoded_imgs.to(DEVICE)
            target_msgs = target_msgs.to(DEVICE)
            
            # 노이즈 적용 (SepMark의 RFNP 참고)
            if model_type == 'tracer':
                # Tracer는 모든 종류의 왜곡에 대응해야 함
                print(f"   배치 {batch_idx+1}: 노이즈 적용 중...") if batch_idx % 5 == 0 else None
                noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=True)
            else:  # detector
                # Detector는 일반 왜곡에는 강인하고 악의적 왜곡에는 취약해야 함
                # 학습 시 50% 확률로 일반/악의적 왜곡 구분
                print(f"   배치 {batch_idx+1}: 노이즈 적용 중...") if batch_idx % 5 == 0 else None
                if torch.rand(1).item() < 0.5:
                    noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=False)
                    target = target_msgs  # 일반 왜곡: 원래 메시지 추출 목표
                else:
                    noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=True)
                    target = torch.zeros_like(target_msgs)  # 악의적 왜곡: 0으로 추출 목표 (랜덤 추측)
            
            # 모델 통과
            print(f"   배치 {batch_idx+1}: 모델 통과 중...") if batch_idx % 5 == 0 else None
            optimizer.zero_grad()
            if model_type == 'tracer':
                extracted_msg = model(noised_imgs)
                loss = criterion(extracted_msg, target_msgs)
                accuracy = calculate_bit_accuracy(extracted_msg, target_msgs)
            else:
                extracted_msg = model(noised_imgs)
                loss = criterion(extracted_msg, target)
                accuracy = calculate_bit_accuracy(extracted_msg, target)
            
            # 역전파 및 옵티마이저 스텝
            print(f"   배치 {batch_idx+1}: 역전파 중...") if batch_idx % 5 == 0 else None
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
            
            if batch_idx % 5 == 0:
                print(f"   배치 {batch_idx+1} 완료: Loss={loss.item():.6f}, Acc={accuracy:.2f}%")
        
        # 에포크 평균 손실 및 정확도 계산
        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # 검증
        print(f"   검증 시작...")
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for val_idx, (encoded_imgs, original_imgs, target_msgs) in enumerate(val_loader):
                if val_idx % 2 == 0:
                    print(f"   검증 배치: {val_idx+1}/{len(val_loader)}")
                    
                encoded_imgs = encoded_imgs.to(DEVICE)
                target_msgs = target_msgs.to(DEVICE)
                
                # 노이즈 적용 (학습과 동일한 방식)
                if model_type == 'tracer':
                    noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=True)
                    extracted_msg = model(noised_imgs)
                    loss = criterion(extracted_msg, target_msgs)
                    accuracy = calculate_bit_accuracy(extracted_msg, target_msgs)
                else:
                    # Detector 검증은 두 가지 경우를 번갈아가며 수행
                    if torch.rand(1).item() < 0.5:
                        noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=False)
                        target = target_msgs
                    else:
                        noised_imgs = noise_pool.random_distortion(encoded_imgs, include_malicious=True)
                        target = torch.zeros_like(target_msgs)
                    
                    extracted_msg = model(noised_imgs)
                    loss = criterion(extracted_msg, target)
                    accuracy = calculate_bit_accuracy(extracted_msg, target)
                
                val_loss += loss.item()
                val_accuracy += accuracy
        
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 에포크 시간 계산
        epoch_time = time.time() - start_time
        
        # 출력
        print(f"📊 에포크 {epoch+1}/{EPOCHS} 완료 | "
              f"Train Loss: {epoch_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Train Acc: {epoch_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"시간: {epoch_time:.2f}초")
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"✅ 새로운 최고 모델 저장 ({model_type})")
        
        # 주기적으로 모델 저장
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = f"./saved_models/{model_type}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
    # 학습 그래프 저장
    print("\n📈 학습 그래프 저장 중...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.capitalize()} Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_type.capitalize()} Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{model_type}_training.png")
    plt.close()
    
    print(f"\n✅ {model_type.capitalize()} 학습 완료")
    return model


# 메인 실행 부분
if __name__ == "__main__":
    import sys
    
    print(f"[시스템 정보] CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[시스템 정보] GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"[설정 정보] 메시지 길이: {MESSAGE_LENGTH}")
    print(f"[설정 정보] 입력 크기: {INPUT_RESIZE}x{INPUT_RESIZE}")
    print(f"[설정 정보] 배치 크기: {BATCH_SIZE}")
    print(f"[설정 정보] 최대 샘플 수: {MAX_SAMPLES}")
    print(f"[설정 정보] 고정 메시지: {FIXED_MESSAGE_STR}")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train_tracer":
            train_decoder('tracer')
        elif command == "train_detector":
            train_decoder('detector')
        elif command == "train_all":
            print("Tracer 학습 시작...")
            train_decoder('tracer')
            print("\nDetector 학습 시작...")
            train_decoder('detector')
        else:
            print("❌ 알 수 없는 명령어입니다. 사용법:")
            print("  python Decoder_train.py train_tracer")
            print("  python Decoder_train.py train_detector")
            print("  python Decoder_train.py train_all")
    else:
        print("[SepMark 디코더 학습 안내]")
        print("다음 명령어 중 하나를 사용하세요:")
        print("  python Decoder_train.py train_tracer   - Tracer 모델 학습")
        print("  python Decoder_train.py train_detector - Detector 모델 학습")
        print("  python Decoder_train.py train_all      - 두 모델 모두 학습")