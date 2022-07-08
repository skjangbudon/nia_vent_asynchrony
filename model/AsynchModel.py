import torch
import torch.nn as nn
import torch.nn.functional as F

class AsynchModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.latent_dim = 30 
        self.kernel_size = 5
        self.pool_size = 4
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, self.kernel_size, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(self.pool_size),
            nn.Conv1d(16, 32, self.kernel_size, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(self.pool_size),
            nn.Conv1d(32, 64, self.kernel_size, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.latent_dim*64),
            nn.Linear(self.latent_dim*64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax()
        )

    def forward(self, x, prt=False):
        if prt: print('input', x.shape)

        x = self.encoder_conv(x)
        if prt: print('encoder_conv', x.shape)
        
        x = self.classifier(x)
        if prt: print('classifier', x.shape)
        
        x = x[:,1]
        return x