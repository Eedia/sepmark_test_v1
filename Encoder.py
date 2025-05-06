# === Encoder_U.py (custom dataset 기반 encoder 학습 with 실시간 상태 출력 + 그래프 주기 저장) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OMP 오류 회피용 설정
import time
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from ConvBlock import ConvBlock
from ResBlock import ResBlock

# === 경로 및 하이퍼파라미터 설정 ===
data_path = "./dataset/val_128"
save_path = "./dataset/saved_model"
os.makedirs(save_path, exist_ok=True)

image_size = 256
message_length = 128
batch_size = 16
learning_rate = 0.0002
beta1 = 0.5
epochs = 100
train_ratio = 0.8

# === Custom Dataset ===
class SimpleImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder, '*.png')) +
                                  glob.glob(os.path.join(folder, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )
    def forward(self, x):
        return self.layer(x)

class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DW_Encoder(nn.Module):
    def __init__(self, message_length, blocks=2, channels=64, attention=None):
        super(DW_Encoder, self).__init__()
        self.conv1 = ConvBlock(3, 16, blocks=blocks)
        self.down1 = Down(16, 32, blocks=blocks)
        self.down2 = Down(32, 64, blocks=blocks)
        self.down3 = Down(64, 128, blocks=blocks)
        self.down4 = Down(128, 256, blocks=blocks)

        self.up3 = UP(256, 128)
        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(128 * 2 + channels, 128, blocks=blocks, attention=attention)

        self.up2 = UP(128, 64)
        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(64 * 2 + channels, 64, blocks=blocks, attention=attention)

        self.up1 = UP(64, 32)
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(32 * 2 + channels, 32, blocks=blocks, attention=attention)

        self.up0 = UP(32, 16)
        self.linear0 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message0 = ConvBlock(1, channels, blocks=blocks)
        self.att0 = ResBlock(16 * 2 + channels, 16, blocks=blocks, attention=attention)

        self.Conv_1x1 = nn.Conv2d(16 + 3, 3, kernel_size=1, stride=1, padding=0)
        self.message_length = message_length

    def forward(self, x, watermark):
        d0 = self.conv1(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(d4)
        expanded_message = self.linear3(watermark).view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]), mode='nearest')
        expanded_message = self.Conv_message3(expanded_message)
        u3 = self.att3(torch.cat((d3, u3, expanded_message), dim=1))

        u2 = self.up2(u3)
        expanded_message = self.linear2(watermark).view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]), mode='nearest')
        expanded_message = self.Conv_message2(expanded_message)
        u2 = self.att2(torch.cat((d2, u2, expanded_message), dim=1))

        u1 = self.up1(u2)
        expanded_message = self.linear1(watermark).view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d1.shape[2], d1.shape[3]), mode='nearest')
        expanded_message = self.Conv_message1(expanded_message)
        u1 = self.att1(torch.cat((d1, u1, expanded_message), dim=1))

        u0 = self.up0(u1)
        expanded_message = self.linear0(watermark).view(-1, 1, self.message_length, self.message_length)
        expanded_message = F.interpolate(expanded_message, size=(d0.shape[2], d0.shape[3]), mode='nearest')
        expanded_message = self.Conv_message0(expanded_message)
        u0 = self.att0(torch.cat((d0, u0, expanded_message), dim=1))

        image = self.Conv_1x1(torch.cat((x, u0), dim=1))
        forward_image = image.clone().detach()
        gap = forward_image.clamp(-1, 1) - forward_image
        return image + gap


# === 학습 루프 ===
def train_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = SimpleImageDataset(data_path, transform=transform)
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DW_Encoder(message_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    criterion = nn.MSELoss()

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        print(f"\n[Epoch {epoch}/{epochs}] Training...")
        for step, (images, _) in enumerate(train_loader, 1):
            images = images.to(device)
            messages = torch.FloatTensor(images.size(0), message_length).uniform_(-0.1, 0.1).to(device)

            encoded = model(images, messages)
            loss = criterion(encoded, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"  [Train][{step}/{len(train_loader)}] Loss: {loss.item():.4f} | Time: {time.time() - start_time:.1f}s")

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, (images, _) in enumerate(val_loader, 1):
                images = images.to(device)
                messages = torch.FloatTensor(images.size(0), message_length).uniform_(-0.1, 0.1).to(device)
                encoded = model(images, messages)
                loss = criterion(encoded, images)
                val_loss += loss.item()

                print(f"  [Val][{step}/{len(val_loader)}] Loss: {loss.item():.4f} | Time: {time.time() - start_time:.1f}s")

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(1 - val_losses[-1])

        duration = time.time() - start_time
        print(f"→ Epoch {epoch} Summary | Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Acc: {val_accs[-1]:.4f}, Time: {duration:.1f}s")

        torch.save(model.state_dict(), os.path.join(save_path, f"encoder_epoch{epoch}.pth"))

        # 중간 시각화 저장
        if epoch % 10 == 0 or epoch == epochs:
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.title(f'Loss Curve (Epoch {epoch})')
            plt.savefig(os.path.join(save_path, f"loss_curve_epoch{epoch}.png"))
            plt.close()

            plt.figure()
            plt.plot(val_accs, label='Validation Accuracy')
            plt.title(f'Validation Accuracy (Epoch {epoch})')
            plt.savefig(os.path.join(save_path, f"val_accuracy_epoch{epoch}.png"))
            plt.close()

    # 최종 시각화 저장
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Final Loss Curve')
    plt.savefig(os.path.join(save_path, "loss_curve.png"))

    plt.figure()
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Final Validation Accuracy')
    plt.savefig(os.path.join(save_path, "val_accuracy.png"))

if __name__ == '__main__':
    train_encoder()
