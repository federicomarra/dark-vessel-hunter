import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# Reparameterization
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Reconstruction Loss
def sequence_loss_fn(
    recon_batch: torch.Tensor,   # (B, T, F)
    x_batch: torch.Tensor        # (B, T, F)
    ) -> torch.Tensor:
    # Standard MSE reconstruction loss per sequence: (B,).
    
    mse = F.mse_loss(recon_batch, x_batch, reduction="none")  # (B,T,F)
    return mse.mean(dim=(1, 2))  # (B,)

# VAE loss = recon + beta * KL
def vae_loss_fn(
    recon_batch: torch.Tensor,
    x_batch: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float
    ) -> Dict[str, torch.Tensor]:
    #    Returns {'total','recon','kl'} — each (B,).
    
    # reconstruction per sequence
    recon = sequence_loss_fn(recon_batch, x_batch)  # (B,)

    # KL divergence for Gaussian
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)  # (B,)

    total = recon + beta * kl
    return {"total": total, "recon": recon, "kl": kl}

# Beta (linear warmup)
def beta_linear_anneal(epoch: int, warmup_epochs: int, beta_end: float) -> float:
    if warmup_epochs <= 0:
        return beta_end
    return float(min(beta_end, beta_end * (epoch / float(warmup_epochs))))



# with the Variational version we make the latent space probabilistic and smoother
# the KL divergence regularizes the latent space
# latent is forced to be near a Gaussian --> better generalization

class LSTMVAEWithShipType(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_shiptypes: int,
        shiptype_emb_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # heads for VAE parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ShipType conditioning 
        self.shiptype_emb = nn.Embedding(num_shiptypes, shiptype_emb_dim)
        self.fc_z_st_to_h = nn.Linear(latent_dim + shiptype_emb_dim, hidden_dim)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    # Encode -> (mu, logvar)
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder(x)          # h_n[-1]: (B, H)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)                # (B, L)
        logvar = self.fc_logvar(h_last)        # (B, L)
        return mu, logvar

    # Decode with teacher forcing 
    def decode(self, x: torch.Tensor, z: torch.Tensor, ship_type_ids: torch.Tensor) -> torch.Tensor:
        st_emb = self.shiptype_emb(ship_type_ids)             # (B, E)
        z_cond = torch.cat([z, st_emb], dim=1)                # (B, L+E)

        h0 = torch.tanh(self.fc_z_st_to_h(z_cond))            # (B, H)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)    # (layers,B,H)
        c0 = torch.zeros_like(h0)

        out, _ = self.decoder(x, (h0, c0))                    # (B,T,H)
        return self.fc_out(out)                               # (B,T,F)

    # Forward -> recon, mu, logvar, z
    def forward(self, x: torch.Tensor, ship_type_ids: torch.Tensor):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(x, z, ship_type_ids)
        return recon, mu, logvar, z
    
