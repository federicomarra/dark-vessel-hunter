from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset_segments_raw import AISSegmentRaw
from vae_lstm import VAE_LSTM
from vae_lstm2 import VAE_LSTM2
from losses import elbo_mse
import torch, yaml

def collate_fn(batch):
    # batch: list of dicts with key "x" (variable length sequences)
    seqs = [b["x"] for b in batch]
    lengths = torch.tensor([s.size(0) for s in seqs])
    x = pad_sequence(seqs, batch_first=True)
    # create mask, for every sequence position that is not padding
    Tmax = x.size(1)
    rng = torch.arange(Tmax).unsqueeze(0)  # (1, Tmax)
    mask = (rng < lengths.unsqueeze(1)).float()  # (B, Tmax)
    return x.float(), mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset of ONE day
    ds_train = AISSegmentRaw(root_dir="ais-data-parquet/2025-11-01")
    ds_val = AISSegmentRaw(root_dir="ais-data-parquet/2025-11-02")
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = VAE_LSTM(in_dim=5, hidden_size=128, z_dim=32).to(device)
    # model = VAE_LSTM2(input_size=5, hidden_size=128, latent_size=32).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    beta0, betaT, warmup_epochs = 0.0, 1.0, 10
    best = float('inf')

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
        
        if val_loss/nval < best:
            best = val_loss/nval
            torch.save({"model": model.state_dict(), "in_dim": 5, "hidden_size": 128, "z_dim": 32}, "runs/best_vae_lstm.pth")
            print(f"  -> saved runs/best_vae_lstm.pt (best={best:.4f})")


if __name__ == "__main__":
    main()

