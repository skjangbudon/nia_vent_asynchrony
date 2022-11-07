import torch
import torch.nn as nn
import torch.nn.functional as F

class AsynchModel(nn.Module):
    def __init__(self, input_dim=2, num_class=2, gap=True, padding_mode = 'zeros'):
        super().__init__()
        self.gap = gap
        self.latent_dim = 30 
        self.kernel_size = 5
        self.pool_size = 4

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, self.kernel_size, padding='same', padding_mode=padding_mode),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(self.pool_size),
            nn.Conv1d(16, 32, self.kernel_size, padding='same', padding_mode=padding_mode),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(self.pool_size),
            nn.Conv1d(32, 32, self.kernel_size, padding='same', padding_mode=padding_mode),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(self.pool_size),
            nn.Conv1d(32, 64, self.kernel_size, padding='same', padding_mode=padding_mode),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            # nn.BatchNorm1d(64),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64 if self.gap else 64*self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_class),
        )

    def forward(self, x, prt=False):
        if prt: print('input', x.shape)

        x = self.encoder_conv(x)
        if prt: print('encoder_conv', x.shape)

        if self.gap:
            x = self.avg_pool(x)
            if prt: print('avg_pool', x.shape)
        else: 
            x = torch.flatten(x, start_dim=1)
            if prt: print('flatten', x.shape)

        x = self.classifier(x)
        if prt: print('classifier', x.shape)
        
        return x
        