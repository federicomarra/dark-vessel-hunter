import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model: LSTM Autoencoder with ShipType conditioning
# ---------------------------------------------------------------------------

class LSTMAutoencoderWithShipType(nn.Module):
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
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        # Ship type embedding
        self.shiptype_emb = nn.Embedding(num_shiptypes, shiptype_emb_dim)

        # Decoder
        self.fc_z_st_to_h = nn.Linear(latent_dim + shiptype_emb_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]            # (B, hidden_dim)
        z = self.fc_latent(h_last)  # (B, latent_dim)
        return z

    def decode(self, x: torch.Tensor, z: torch.Tensor, ship_type_ids: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)  # teacher forcing
        z: (B, latent_dim)
        ship_type_ids: (B,)
        """
        st_emb = self.shiptype_emb(ship_type_ids)         # (B, emb_dim)
        z_cond = torch.cat([z, st_emb], dim=1)           # (B, latent+emb)

        h0 = torch.tanh(self.fc_z_st_to_h(z_cond))       # (B, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        decoder_input = x  # you could add dropout here if desired
        out, _ = self.decoder(decoder_input, (h0, c0))   # (B, T, hidden_dim)
        recon = self.fc_out(out)                         # (B, T, F)
        return recon

    def forward(self, x: torch.Tensor, ship_type_ids: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(x, z, ship_type_ids)
        return recon, z