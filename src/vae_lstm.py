import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=32, dropout: float=0.1):
        """
        LSTM encoder for the VAE.

        Args:
            in_dim:       Input feature dimension (F).
            hidden_size:  LSTM hidden size (H).
            num_layers:   Number of stacked LSTM layers.
            z_dim:        Latent dimension.
            dropout:      Dropout probability between LSTM layers (only used if num_layers > 1).
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            in_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer to map the final hidden state to the mean (mu) of the latent distribution
        self.hidden_to_mu = nn.Linear(hidden_size, z_dim)
        
        # Linear layer to map the final hidden state to the log-variance (logvar) of the latent distribution
        self.hidden_to_logvar = nn.Linear(hidden_size, z_dim)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """
        Args:
            x:    Input sequence, shape (B, T, F).
            mask: Valid-timestep mask, shape (B, T).

        Returns:
            mu:     Latent mean, shape (B, z_dim).
            logvar: Latent log-variance, shape (B, z_dim).
        """
        out, _ = self.lstm(x) 
        
        # Masked Average Pooling over time
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)        # Shape: (B, 1)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom      # Shape: (B, H)
        
        # Project the pooled summary vector to mu and logvar
        mu = self.hidden_to_mu(pooled)                      # Shape: (B, z_dim)
        logvar = self.hidden_to_logvar(pooled)              # Shape: (B, z_dim)
        
        return mu, logvar


class DecoderLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, num_layers: int=1, z_dim: int=32, dropout: float=0.1):
        """
        LSTM decoder for the VAE.

        At each time step, the decoder receives:
            - previous ground-truth features (teacher forcing)
            - the latent vector z

        Args:
            in_dim:       Output feature dimension (must match encoder input dim).
            hidden_size:  LSTM hidden size.
            num_layers:   Number of LSTM layers.
            z_dim:        Latent dimension.
            dropout:      Dropout probability.
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim

        self.lstm = nn.LSTM(
            input_size=in_dim + z_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.z_to_hidden = nn.Linear(z_dim, hidden_size)
        self.hidden_to_out = nn.Linear(hidden_size, in_dim)

    def forward(self, x:torch.Tensor, z:torch.Tensor):
        """
        Args:
            x: Ground-truth sequence, shape (B, T, F).
            z: Latent code, shape (B, z_dim).

        Returns:
            x_hat: Reconstructed sequence, shape (B, T, F).
        """
        B, T, F = x.shape

        start = torch.zeros((B, 1, F), device=x.device, dtype=x.dtype)          # Shape: (B, 1, F)
        x_shifted = torch.cat([start, x[:, :-1, :]], dim=1)                     # Shape: (B, T, F)

        # repeat 'z' for each time step
        z_rep = z.unsqueeze(1).expand(B, T, self.z_dim)                         # Shape: (B, T, Z)

        # Concatenate features and latent code
        dec_in = torch.cat([x_shifted, z_rep], dim=-1)                          # Shape: (B, T, F + Z)

        # initialize hidden and cell states from 'z'
        h0 = torch.tanh(self.z_to_hidden(z)).unsqueeze(0)                       # Shape: (1, B, H)
        c0 = torch.zeros_like(h0)                                               # Shape: (1, B, H)

        out, _ = self.lstm(dec_in, (h0, c0))                                    # out shape: (B, T, H)
        x_hat = self.hidden_to_out(out)                                         # x_hat shape: (B, T, F)
        
        return x_hat
    

class VAE_LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int=128, z_dim: int=32, num_layers: int=1, dropout: float=0.1):
        """
        Full VAE-LSTM model: encoder + decoder.
        """
        super().__init__()
        self.encoder = EncoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)
        self.decoder = DecoderLSTM(in_dim, hidden_size, num_layers, z_dim, dropout)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """
        Args:
            x:    Input sequence, shape (B, T, F).
            mask: Valid-timestep mask, shape (B, T).

        Returns:
            x_hat: Reconstructed sequence, shape (B, T, F).
            mu:    Latent mean, shape (B, z_dim).
            logvar:Latent log-variance, shape (B, z_dim).
        """
        mu, logvar = self.encoder(x, mask) 
        std = torch.exp(0.5 * logvar)       # Calculate standard deviation
        eps = torch.randn_like(std)         # Sample from N(0, 1)
        z = mu + eps * std                  # Sample 'z' from N(mu, std)
        
        x_hat = self.decoder(x, z)          # x_hat shape: (B, T, in_Sdim)
        return x_hat, mu, logvar