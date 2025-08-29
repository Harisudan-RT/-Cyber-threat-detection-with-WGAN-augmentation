import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=32, output_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
