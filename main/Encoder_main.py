import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from Encoder import DW_Encoder  # 반드시 같은 구조여야 함

# ===== 사용자 정의 =====
model_path = "./dataset/saved_model/encoder_epoch100.pth"
input_folder = "./sample"
output_folder = "./test_output_encoded" 
# input_folder = "./dataset/val_128"  
# output_folder = "./test_output_encoded_val_128" #디코더 학습을 위한 인코딩된 데이터셋 저장을 위한 폴더. 
message_length = 128
input_resize = 256
fixed_message_str = "hello world"
max_workers = 4
target_psnr = 40.0  # PSNR 목표 (40dB 이상)

# ===== 1. 모델 불러오기 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DW_Encoder(message_length).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== 2. 고정 메시지 생성 =====
def make_fixed_message(text, length):
    text_bytes = text.encode('utf-8')
    text_tensor = torch.tensor(list(text_bytes), dtype=torch.float32)

    if text_tensor.numel() >= length:
        message = text_tensor[:length]
    else:
        pad_size = length - text_tensor.numel()
        message = torch.cat([text_tensor, torch.zeros(pad_size)], dim=0)

    # Center around 0, scale small
    message = (message - 128) / 128 * 0.2  # -0.1~0.1
    return message.unsqueeze(0)

fixed_message = make_fixed_message(fixed_message_str, message_length).to(device)

# ===== 3. Transform 정의
transform = transforms.Compose([
    transforms.Resize((input_resize, input_resize)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 결과 폴더 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# ===== 4. PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# ===== 5. 하나의 파일 처리하는 함수
def encode_image(filename):
    try:
        if not filename.lower().endswith(".png"):
            return
        
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f"encoded_{filename}")

        # 원본 이미지 로드
        original_image = Image.open(input_image_path).convert("RGB")
        original_size = original_image.size
        
        small_image = transform(original_image)
        tensor_image = small_image.unsqueeze(0).to(device)

        # 디노말라이즈 준비
        denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

        # ===== Adaptive 삽입 시작 =====
        scaling = 1.0
        max_try = 8  # 시도 횟수 늘림

        for attempt in range(max_try):
            adjusted_message = fixed_message * scaling

            with torch.no_grad():
                encoded = model(tensor_image, adjusted_message)

            # 균등한 RGB 채널 삽입 적용
            encoded = encoded.squeeze(0)
            tensor_image_norm = tensor_image.squeeze(0)
            perturbed = tensor_image_norm + (encoded - tensor_image_norm) * 0.5  # 차이값을 약하게
            perturbed = perturbed.clamp(-1, 1)

            # PSNR 체크
            decoded = denormalize(perturbed.cpu()).clamp(0, 1)
            original_decoded = denormalize(tensor_image_norm.cpu()).clamp(0, 1)

            psnr = calculate_psnr(decoded, original_decoded)

            if psnr >= target_psnr:
                break
            else:
                scaling *= 0.7  # 더 공격적으로 줄이기 (70%만 남기기)

        # 업샘플링
        output_tensor = F.interpolate(decoded.unsqueeze(0), size=original_size[::-1], mode='bilinear', align_corners=False).squeeze(0)
        output_image = transforms.ToPILImage()(output_tensor)
        output_image.save(output_image_path)

        print(f"✅ {filename} 워터마크 완료 (PSNR={psnr:.2f}dB) → {output_image_path}")
    
    except Exception as e:
        print(f"❌ 실패 ({filename}): {e}")

# ===== 6. 모든 파일 병렬 처리
if __name__ == "__main__":
    files = os.listdir(input_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(encode_image, files)
