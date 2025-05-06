import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# SepMark 참고: 하나의 인코더와 두 개의 분리된 디코더(Tracer, Detector)

class ConvINRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvINRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvINRelu(in_channels, out_channels),
            ConvINRelu(out_channels, out_channels)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEResidualBlock, self).__init__()
        self.conv1 = ConvINRelu(in_channels, out_channels)
        self.conv2 = ConvINRelu(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return F.relu(out)

# 디코더 클래스: Tracer (강건한 디코더)
class DW_Tracer(nn.Module):
    def __init__(self, message_length=128):
        super(DW_Tracer, self).__init__()
        self.message_length = message_length
        
        # 인코더와 비슷한 다운샘플링 구조
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        # 병목 SE-블록
        self.bottleneck = SEResidualBlock(512, 512)
        
        # 업샘플링 경로
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        # 메시지 추출 레이어
        self.residual = nn.Conv2d(64, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((message_length, message_length))
        self.final = nn.Conv2d(1, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.message_fc = nn.Linear(message_length * message_length, message_length)
        
    def forward(self, x):
        # 이미지 인코딩
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 병목
        x5 = self.bottleneck(x5)
        
        # 업샘플링과 스킵 연결
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 메시지 추출
        x = self.residual(x)
        x = self.avgpool(x)
        x = self.final(x)
        x = self.flatten(x)
        x = self.message_fc(x)
        
        return x

# 디코더 클래스: Detector (준강건한 디코더)
class DW_Detector(nn.Module):
    def __init__(self, message_length=128):
        super(DW_Detector, self).__init__()
        self.message_length = message_length
        
        # Tracer와 유사하지만 약간 다른 구조로 설계
        self.inc = DoubleConv(3, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.down3 = Down(192, 384)
        
        # SE 블록 (주의 메커니즘)
        self.bottleneck = SEResidualBlock(384, 384)
        
        # 업샘플링 경로
        self.up1 = Up(576, 192)
        self.up2 = Up(288, 96)
        self.up3 = Up(144, 48)
        
        # 메시지 추출 레이어
        self.residual = nn.Conv2d(48, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((message_length, message_length))
        self.final = nn.Conv2d(1, 1, kernel_size=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.message_fc = nn.Linear(message_length * message_length, message_length)
        
    def forward(self, x):
        # 이미지 인코딩
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 병목
        x4 = self.bottleneck(x4)
        
        # 업샘플링과 스킵 연결
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # 메시지 추출
        x = self.residual(x)
        x = self.avgpool(x)
        x = self.final(x)
        x = self.flatten(x)
        x = self.message_fc(x)
        
        return x
