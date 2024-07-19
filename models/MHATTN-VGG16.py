import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        x = x.permute(2, 0, 1)  # (seq_len, batch, features)
        attn_output, _ = self.mha(x, x, x)
        return attn_output.permute(1, 2, 0)  # (batch, features, seq_len)

class MHATTNVGG16(nn.Module):
    def __init__(self, num_classes, num_heads=8):
        super(MHATTNVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.features = self.vgg16.features[:-1]  # Remove last max pooling
        
        self.attention = MultiHeadAttention(512, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Usage
model = MHATTNVGG16(num_classes=3)  # 3 classes for buy, sell, hold