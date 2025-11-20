import os
import torch
from torch.utils.data import DataLoader

from dataset_segments_raw import AISSegmentRaw
from vae_lstm import VAE_LSTM
from vae_lstm2 import VAE_LSTM2
from losses import elbo_mse


def collate_fixed_length(batch):
    """
    Collate function for fixed-length windows.

    Each item in the batch is a dict with key 'x':
        x: (T, F) tensor (already normalized).

    We stack them to get:
        X: (B, T, F)
    and create a mask of ones with shape (B, T).
    """
    x_list = [b["x"] for b in batch]       # list of (T, F)
    x = torch.stack(x_list, dim=0)         # (B, T, F)
    B, T, _ = x.shape
    mask = torch.ones(B, T, dtype=torch.float32)
    return x.float(), mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    # Adjust these dates depending on your data availability    
    train_dates = ["2025-11-01"]
    val_dates = ["2025-11-02"]

    ds_train = AISSegmentRaw(root_dir="ais-data/parquet", window_size=30, min_points=30, allowed_dates=train_dates)
    ds_val = AISSegmentRaw(root_dir="ais-data/parquet", window_size=30, min_points=30, allowed_dates=val_dates)
    dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, collate_fn=collate_fixed_length)
    dl_val = DataLoader(ds_val, batch_size=50, shuffle=False, collate_fn=collate_fixed_length)
    in_dim = ds_train.num_features

    # Model & Optimizer
    model = VAE_LSTM(in_dim=in_dim, hidden_size=128, z_dim=32).to(device)
    # model = VAE_LSTM2(input_size=in_dim, hidden_size=128, latent_size=32).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    beta0, betaT, warmup_epochs = 0.0, 1.0, 10
    best_val = float('inf')

    # ------
    # Training Loop
    # ------
    for epoch in range(1, num_epochs + 1):
        model.train()
        beta = beta0 + (betaT - beta0) * min(epoch / warmup_epochs, 1.0)

        train_loss = train_rec = train_kl = 0.0

        for batch_idx, (x, mask) in enumerate(dl_train):
            x, mask = x.to(device), mask.to(device)

            if epoch == 1 and batch_idx == 0:
                valid = x[~torch.isnan(x)]
                print("DEBUG x stats:",
                    "min", valid.min().item() if valid.numel() > 0 else None,
                    "max", valid.max().item() if valid.numel() > 0 else None,
                    "has_nan", torch.isnan(x).any().item())
                
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, mask)
            loss, parts = elbo_mse(x_hat, x, mask, mu, logvar, beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() 
            train_rec += parts['reconstruction'].item() 
            train_kl += parts['kl_divergence'].item() 

        ntr = len(dl_train)

        # Validation Loop
        model.eval()
        val_loss = val_rec = val_kl = 0.0

        with torch.no_grad():
            for x, mask in dl_val:
                x, mask = x.to(device), mask.to(device)

                x_hat, mu, logvar = model(x, mask)
                loss, parts = elbo_mse(x_hat, x, mask, mu, logvar, beta=1.0)

                val_loss += loss.item() 
                val_rec += parts['reconstruction'].item() 
                val_kl += parts['kl_divergence'].item()

        nval = len(dl_val)

        print(f"Epoch {epoch:02d}/{num_epochs:02d}| β {beta:.2f} | "
              f"Tr {train_loss/ntr:.4f} (R {train_rec/ntr:.4f} KL {train_kl/ntr:.4f}) | "
              f"Va {val_loss/nval:.4f} (R {val_rec/nval:.4f} KL {val_kl/nval:.4f})")
        
        cur_val = val_loss / nval
        if cur_val < best_val:
            best_val = cur_val
            # Create folder if it doesn't exist
            os.makedirs("runs", exist_ok=True)
            ckpt = {
                "model": model.state_dict(),
                "in_dim": in_dim,
                "hidden_size": 128,
                "z_dim": 32,
                "mean": ds_train.mean,   # normalization stats (for reference)
                "std": ds_train.std,
            }
            torch.save(ckpt, "runs/best_vae_lstm_chunks.pth")
            print(f"  -> saved runs/best_vae_lstm_chunks.pth (best Val={best_val:.4f})")

if __name__ == "__main__":
    main()

