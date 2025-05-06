import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from io import BytesIO

# SepMark의 Random Forward Noise Pool (RFNP) 구현
class RandomForwardNoisePool:
    def __init__(self):
        self.common_distortions = [
            self.identity,
            self.jpeg_compression,
            self.resize,
            self.gaussian_blur,
            self.median_blur,
            self.brightness_adjust,
            self.contrast_adjust,
            self.saturation_adjust,
            self.hue_adjust,
            self.dropout,
            self.salt_pepper,
            self.gaussian_noise
        ]
        
        self.malicious_distortions = [
            self.simswap_simulation,    # 실제 SimSwap 대신 시뮬레이션
            self.ganimation_simulation, # 실제 GANimation 대신 시뮬레이션
            self.stargan_simulation     # 실제 StarGAN 대신 시뮬레이션
        ]
    
    def random_distortion(self, images, include_malicious=True):
        """무작위로 왜곡 함수를 선택하여 이미지에 적용"""
        if include_malicious and torch.rand(1).item() < 0.3:  # 30% 확률로 악의적 왜곡 선택
            distortion_fn = np.random.choice(self.malicious_distortions)
        else:
            distortion_fn = np.random.choice(self.common_distortions)
        
        return distortion_fn(images)
    
    def identity(self, images):
        """항등 변환 (왜곡 없음)"""
        return images
    
    def jpeg_compression(self, images, quality=50):
        """JPEG 압축 시뮬레이션"""
        bs, c, h, w = images.shape
        result = torch.zeros_like(images)
        
        for i in range(bs):
            img = (images[i].permute(1, 2, 0).cpu().detach() + 1) / 2 * 255
            img = img.numpy().astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            
            compressed_img = Image.open(buffer)
            compressed_tensor = torch.from_numpy(np.array(compressed_img)).permute(2, 0, 1) / 127.5 - 1
            result[i] = compressed_tensor.to(images.device)
        
        return result
    
    def resize(self, images):
        """리사이징 시뮬레이션 (다운샘플 후 업샘플)"""
        scale_factor = torch.rand(1).item() * 0.5 + 0.5  # 0.5~1.0 사이의 스케일
        h, w = images.shape[2:]
        
        downscaled = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        upscaled = F.interpolate(downscaled, size=(h, w), mode='bilinear', align_corners=False)
        
        return upscaled
    
    def gaussian_blur(self, images, kernel_size=3, sigma=2.0):
        """가우시안 블러 시뮬레이션"""
        bs, c, h, w = images.shape
        result = torch.zeros_like(images)
        
        # 가우시안 커널 생성
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        center = kernel_size // 2
        x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
        x, y = x.to(images.device), y.to(images.device)
        
        kernel = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
        
        # Padding
        padded = F.pad(images, (center, center, center, center), mode='replicate')
        
        # 채널별 컨볼루션
        for i in range(bs):
            for j in range(c):
                result[i, j] = F.conv2d(padded[i:i+1, j:j+1], kernel[j:j+1], padding=0)
        
        return result
    
    def median_blur(self, images, kernel_size=3):
        """미디안 블러 시뮬레이션"""
        # 이 함수는 연산량이 많기 때문에 간소화된 버전 사용
        # 실제로는 OpenCV나 kornia 라이브러리를 사용하는 것이 효율적
        bs, c, h, w = images.shape
        result = images.clone()
        
        # 미디안 필터를 더 간단하게 구현: 다운샘플링 후 업샘플링으로 대체
        result = F.interpolate(result, scale_factor=0.5, mode='bilinear', align_corners=False)
        result = F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False)
        
        return result
    
    def brightness_adjust(self, images, factor=None):
        """밝기 조정"""
        if factor is None:
            factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8~1.2 사이의 밝기 변화
        
        return images * factor
    
    def contrast_adjust(self, images, factor=None):
        """대비 조정"""
        if factor is None:
            factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8~1.2 사이의 대비 변화
        
        mean = torch.mean(images, dim=[1, 2, 3], keepdim=True)
        return (images - mean) * factor + mean
    
    def saturation_adjust(self, images, factor=None):
        """채도 조정 (간단히 구현)"""
        if factor is None:
            factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8~1.2 사이의 채도 변화
        
        gray = torch.mean(images, dim=1, keepdim=True)
        return images * factor + gray * (1 - factor)
    
    def hue_adjust(self, images, factor=None):
        """색조 조정 (단순화된 구현)"""
        if factor is None:
            factor = torch.rand(1).item() * 0.2 - 0.1  # -0.1~0.1 사이의 색조 변화
        
        # RGB -> HSV -> RGB 변환 대신 간단한 채널 섞기
        r, g, b = images.chunk(3, dim=1)
        
        new_r = r * (1 - factor) + g * factor
        new_g = g * (1 - factor) + b * factor
        new_b = b * (1 - factor) + r * factor
        
        return torch.cat([new_r, new_g, new_b], dim=1)
    
    def dropout(self, images, p=0.05):
        """랜덤 드롭아웃"""
        mask = (torch.rand_like(images) > p).float()
        return images * mask
    
    def salt_pepper(self, images, p=0.02):
        """소금-후추 노이즈"""
        noise = torch.rand_like(images)
        salt = (noise < p/2).float()
        pepper = (noise > (1 - p/2)).float()
        
        return images * (1 - salt - pepper) + salt - pepper
    
    def gaussian_noise(self, images, std=0.05):
        """가우시안 노이즈"""
        noise = torch.randn_like(images) * std
        return images + noise
    
    # 악의적 왜곡 시뮬레이션
    def simswap_simulation(self, images):
        """단순화된 SimSwap 효과 시뮬레이션"""
        return self.gaussian_blur(images) + self.gaussian_noise(images, std=0.1)
    
    def ganimation_simulation(self, images):
        """단순화된 GANimation 효과 시뮬레이션"""
        return self.brightness_adjust(self.contrast_adjust(images), factor=1.1)
    
    def stargan_simulation(self, images):
        """단순화된 StarGAN 효과 시뮬레이션"""
        return self.saturation_adjust(self.hue_adjust(images, factor=0.15), factor=1.2)
