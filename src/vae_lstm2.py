import torch
import torch.nn as nn

class EncoderLSTM2(nn.Module):
    def __init__(self, in_dim, hidden_size=128, z_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.mu = nn.Linear(hidden_size, z_dim)
        self.logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x, mask):
        out, _ = self.lstm(x)
        # masked mean over time 
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        mu = self.mu(pooled)
        logvar = self.logvar(pooled)
        return mu, logvar


class DecoderLSTM2(nn.Module):
    def __init__(self, out_dim, hidden_size=128, z_dim=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.lstm = nn.LSTM(
            input_size=z_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, out_dim)
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)

    def forward(self, x, z):
        B, T, _ = x.shape
        z_seq = z.unsqueeze(1).expand(B, T, self.z_dim)
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(z_seq, (h0, c0))
        x_hat = self.fc(out)
        return x_hat


class VAE_LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size=128, latent_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.encoder = EncoderLSTM2(input_size, hidden_size, latent_size, num_layers, dropout)
        self.decoder = DecoderLSTM2(input_size, hidden_size, latent_size, num_layers, dropout)

    def forward(self, x, mask):
        mu, logvar = self.encoder(x, mask)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(x, z)
        return x_hat, mu, logvar
