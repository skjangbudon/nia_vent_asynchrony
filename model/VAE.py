import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = 30 # encoder output
        self.zDim = 128
        self.kernel_size = 5
        self.pool_size = 4

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_dim, 16, self.kernel_size, padding='same'),
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

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, self.kernel_size, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.pool_size, mode='linear'),
            nn.ConvTranspose1d(32, 16, self.kernel_size, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.pool_size, mode='linear'),
            nn.ConvTranspose1d(16, input_dim, self.kernel_size, padding=2),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )
        self.encFC1 = nn.Linear(self.latent_dim, self.zDim)
        self.encFC2 = nn.Linear(self.latent_dim, self.zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(self.zDim, self.latent_dim)

    def encoder(self, x, prt=False):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        if prt: print('input', x.shape)

        x = self.encoder_conv(x)
        if prt: print('encoder_conv', x.shape)
        
        x = x.view(-1, self.latent_dim)
        if prt: print('view', x.shape)

        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, prt=False):
        if prt: print('z', z.shape)

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        if prt: print('decFC1', x.shape)

        x = x.view(-1, 64, self.latent_dim)
        if prt: print('view', x.shape)

        # x = torch.sigmoid(self.decoder_conv(x))
        x = self.decoder_conv(x)
        if prt: print('decoder_conv', x.shape)

        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar