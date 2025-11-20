# Compute the ELBO loss for a VAE with an MSE reconstruction term:
# ELBO = reconstruction_loss + beta * KL

import torch

def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    ''' Mean Squared Error over time and features, masked by valid timesteps.
    Args:
        x_hat: Reconstructed sequence, shape (B, T, F).
        x:     Original sequence, shape (B, T, F).
        mask:  Binary mask, shape (B, T), where 1 = valid timestep, 0 = padding.
        eps:   Small constant to avoid division by zero (not strictly needed here).

    Returns:
        Scalar tensor: masked MSE. '''
    err2 = (x_hat - x) ** 2             # (B, T, F)
    err2 = err2.mean(dim=-1)            # (B, T) average over features
    err2 = err2 * mask                  # zero out padded timesteps

    denom = mask.sum().clamp_min(1.0)
    return err2.sum() / denom


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    '''
    KL divergence between N(mu, sigma^2) and N(0, 1) for a diagonal Gaussian.

    Args:
        mu:     Mean of latent distribution, shape (B, z_dim).
        logvar: Log-variance of latent distribution, shape (B, z_dim).

    Returns:
        Scalar tensor: average KL divergence over the batch. '''
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=-1)  # (B,)
    return kl.mean()                    # scalar


def elbo_mse(x_hat, x, mask, mu, logvar, beta: float=1.0) -> torch.Tensor:
    '''
    Compute the ELBO loss for a VAE with an MSE reconstruction term:

        ELBO = reconstruction_loss + beta * KL

    Args:
        x_hat: Reconstructed sequence (B, T, F).
        x:     Ground-truth sequence (B, T, F).
        mask:  Valid-timestep mask (B, T).
        mu:    Latent mean (B, z_dim).
        logvar:Latent log-variance (B, z_dim).
        beta:  Weight for the KL term (beta-VAE style warmup).

    Returns:
        loss:  Scalar tensor (ELBO).
        parts: Dict with "reconstruction" and "kl_divergence" (both detached).
    '''
    rec_loss = masked_mse(x_hat, x, mask)
    kl_loss = kl_standard_normal(mu, logvar)
    loss = rec_loss + beta * kl_loss
    parts = {'reconstruction': rec_loss.detach(), 'kl_divergence': kl_loss.detach()}
    return loss, parts
