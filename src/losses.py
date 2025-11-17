# Loss ELBO per VAE-LSTM (reconstruction + KL divergence)
# Reconstruction = MSE weighted by mask on valid time steps

import torch

def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    err2 = (x_hat - x) ** 2               # (B, T, F)
    err2 = err2.mean(dim=-1)        # (B, T)
    err2 = err2 * mask              # (B, T)
    denom = mask.sum().clamp_min(1.0)
    return err2.sum() / denom

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mu, logvar: (B, z_dim)
    # KL divergence between N(mu, var) and N(0, 1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=-1)  # (B,)
    return kl.mean()  # scalar

def elbo_mse(x_hat, x, mask, mu, logvar, beta: float=1.0) -> torch.Tensor:
    rec_loss = masked_mse(x_hat, x, mask)
    kl_loss = kl_standard_normal(mu, logvar)
    loss = rec_loss + beta * kl_loss
    parts = {'reconstruction': rec_loss.detach(), 'kl_divergence': kl_loss.detach()}
    return loss, parts

