# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch import flatten
from torch.nn import functional as F

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class CNN(nn.Module):
    def __init__(self, in_features, nclass=2, **kwargs):
        super(CNN, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=40),
            )
        
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, nclass)
        self.bn1 = nn.BatchNorm1d(nclass)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()  
            x = x - torch.mean(x, dim=-1, keepdim=True)     
        x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        return x


if __name__ == "__main__":
    model = CNN(in_features=1, nclass=2)
    x = torch.Tensor(np.random.rand(32, 32000))
    y = model(x)
    print(y.shape)
