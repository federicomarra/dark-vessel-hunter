# Main training script

# File imports
import config
import ais_query

# Imports
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
import math

VERBOSE_MODE = config.VERBOSE_MODE

FOLDER_NAME = config.FOLDER_NAME
folder_path = Path(FOLDER_NAME)
parquet_folder_path = folder_path / "parquet"

SEGMENT_MAX_LENGTH = config.SEGMENT_MAX_LENGTH
NUMERIC_COLS = config.NUMERIC_COLS
PRE_PROCESSING_DF_PATH = config.PRE_PROCESSING_DF_PATH

class AISTrajectoryDataset(Dataset):
    def __init__(self, df_seq: pd.DataFrame):
        # Save a copy and reset index
        self.df = df_seq.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Sequence: list-of-lists -> numpy -> tensor (30, F)
        seq = np.array(row["Sequence"], dtype=np.float32)
        x = torch.from_numpy(seq)  # shape (30, F)

        # ShipTypeID: integer -> tensor (for future embedding)
        ship_type_id = int(row["ShipTypeID"])
        ship_type_id = torch.tensor(ship_type_id, dtype=torch.long)

        return x, ship_type_id



class LSTMVAEWithShipType(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 num_shiptypes: int,
                 shiptype_emb_dim: int = 8,
                 num_layers: int = 1):
        super().__init__()

        # ----- ENCODER -----
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----- SHIP TYPE EMBEDDING -----
        self.shiptype_emb = nn.Embedding(num_shiptypes, shiptype_emb_dim)

        # ----- DECODER -----
        # we'll concatenate z and shiptype_emb, then map to initial hidden state
        self.fc_z_st_to_h = nn.Linear(latent_dim + shiptype_emb_dim, hidden_dim)

        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    # ---- core VAE ops ----

    def encode(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.encoder(x)
        h_T = h_n[-1]           # (B, hidden_dim)
        mu = self.fc_mu(h_T)    # (B, latent_dim)
        logvar = self.fc_logvar(h_T)  # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std   # (B, latent_dim)

    def decode(self, x, z, ship_type_ids):
        """
        x: (B, T, F)  - original sequence (teacher forcing)
        z: (B, latent_dim)
        ship_type_ids: (B,)
        """
        # ship type embedding
        st_emb = self.shiptype_emb(ship_type_ids)       # (B, shiptype_emb_dim)

        # concatenate z and ship type embedding
        z_cond = torch.cat([z, st_emb], dim=1)          # (B, latent_dim+emb_dim)

        # map to initial hidden state of decoder
        h0 = torch.tanh(self.fc_z_st_to_h(z_cond))      # (B, hidden_dim)
        h0 = h0.unsqueeze(0)                            # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)                       # (1, B, hidden_dim)

        out, _ = self.decoder(x, (h0, c0))              # (B, T, hidden_dim)
        recon = self.fc_out(out)                        # (B, T, F)
        return recon

    def forward(self, x, ship_type_ids):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(x, z, ship_type_ids)
        return recon, mu, logvar




# Loss functions: reconstruction + KL
def reconstruction_loss(x, recon_x):
    # MSE over (B, T, F)
    return torch.mean((x - recon_x) ** 2)

def kl_loss(mu, logvar):
    # KL(q(z|x) || N(0, I)) per batch
    # mu, logvar: (B, latent_dim)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()



def plot_original_vs_recon(
    model,
    dataloader,
    device,
    feature_names,
    num_examples: int = 3
):
    model.eval()
    feature_names = list(feature_names)
    
    with torch.no_grad():
        # get a single batch
        x_batch, ship_type_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)              # (B, T, F)
        ship_type_batch = ship_type_batch.to(device)
        
        recon_batch, mu, logvar = model(x_batch, ship_type_batch)
        
        x_batch = x_batch.cpu().numpy()
        recon_batch = recon_batch.cpu().numpy()
    
    B, T, F = x_batch.shape
    num_examples = min(num_examples, B)
    
    for i in range(num_examples):
        orig = x_batch[i]   # (T, F)
        rec  = recon_batch[i]
        
        print(f"\nPlotting example {i} (sequence index {i})")
        
        # grid layout for subplots
        n_features = F
        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
        
        time_steps = np.arange(T)
        
        for f_idx in range(n_features):
            row = f_idx // n_cols
            col = f_idx % n_cols
            
            ax = axes[row][col]
            
            fname = feature_names[f_idx] if f_idx < len(feature_names) else f"feat_{f_idx}"
            
            ax.plot(time_steps, orig[:, f_idx], label="original")
            ax.plot(time_steps, rec[:, f_idx], linestyle="--", label="reconstructed")
            
            ax.set_title(fname)
            ax.set_xlabel("timestep")
            ax.set_ylabel("value")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # hide any empty subplots if F is odd
        for f_idx in range(n_features, n_rows * n_cols):
            row = f_idx // n_cols
            col = f_idx % n_cols
            fig.delaxes(axes[row][col])
        
        fig.tight_layout()
        plt.show()
        
        


def main_training():
    # --- DATA LOADING PREPROCESSING ---
    df = pd.read_parquet(PRE_PROCESSING_DF_PATH)
    NAV_ONEHOT_COLS = [c for c in df.columns if c.startswith("NavStatus_")]
    FEATURE_COLS = NUMERIC_COLS + NAV_ONEHOT_COLS
    sequences = []
    groups = df.groupby("Segment_nr")
    
    for seg_id, g in groups:
        g = g.sort_values("Timestamp")  # or Timestamp; important is temporal order
        if len(g) != config.SEGMENT_MAX_LENGTH:
            continue  # safety: skip anomalous segments

        X = g[FEATURE_COLS].to_numpy(dtype=float)  # shape (30, F_tot)
        ship_type_id = int(g["ShipTypeID"].iloc[0])
        mmsi = g["MMSI"].iloc[0]

        sequences.append({
            "Segment_nr": seg_id,
            "MMSI": mmsi,
            "ShipTypeID": ship_type_id,
            "Sequence": X,
        })
    df_seq = pd.DataFrame(sequences)
    
    
    dataset = AISTrajectoryDataset(df_seq)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # sanity check:
    x_batch, st_batch = next(iter(dataloader))
    print(x_batch.shape)   # -> [batch_size, 30, F]
    print(st_batch.shape)  # -> [batch_size]

    # --- MODEL SETUP ----
    # Instantiating the model
    num_shiptypes = df_seq["ShipTypeID"].nunique()
    print("num_shiptypes =", num_shiptypes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_x, _ = dataset[0]
    T, F = sample_x.shape

    model = LSTMVAEWithShipType(
        input_dim=F,
        hidden_dim=config.HIDDEN_DIM,        # tune this
        latent_dim=config.LATENT_DIM,        # tune this
        num_shiptypes=num_shiptypes,
        shiptype_emb_dim=8,   # small embedding is enough
        num_layers=config.NUM_LAYERS,            # tune this
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    beta = config.BETA  # weight for KL term, you can tune this

    # Training loop: VAE + ship type
    num_epochs = config.EPOCHS

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0

        for x, ship_type_id in dataloader:
            x = x.to(device)                      # (B, T, F)
            ship_type_id = ship_type_id.to(device)

            recon, mu, logvar = model(x, ship_type_id)

            rec = reconstruction_loss(x, recon)
            kld = kl_loss(mu, logvar)
            loss = rec + beta * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_rec  += rec.item() * x.size(0)
            total_kl   += kld.item() * x.size(0)

        N = len(dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- loss: {total_loss/N:.6f} "
            f"- rec: {total_rec/N:.6f} "
            f"- kl: {total_kl/N:.6f}"
        )
    
    # Performance check
    # Real vs reconstructed
    features_names = [
        "Latitude",
        "Longitude",
        "SOG",
        "COG",
        "DeltaT",
        "NavStatus_0",
        "NavStatus_1",
        "NavStatus_2",
        "NavStatus_3",
        "NavStatus_4",
        "NavStatus_5",
        "NavStatus_6",
    ]
    plot_original_vs_recon(
        model=model,
        dataloader=dataloader,
        device=device,
        feature_names=features_names,
        num_examples=3,   # how many sequences to inspect from the batch
    )
    
    
    
    
    
if __name__ == "__main__":
    main_training()