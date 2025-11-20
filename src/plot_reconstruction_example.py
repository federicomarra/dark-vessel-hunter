#!/usr/bin/env python3
"""
Plot original vs reconstructed trajectory features for a single window
of a given MMSI / date / segment.

Example usage:

    python src/plot_reconstruction_example.py \
        --ckpt runs/best_vae_lstm_chunks.pth \
        --root-dir ais-data/parquet \
        --mmsi 209275000 \
        --date 2025-11-01 \
        --segment-id 0 \
        --window-num 0
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset_segments_raw import AISSegmentRaw
from vae_lstm import VAE_LSTM


def find_windows_for_segment(
    dataset: AISSegmentRaw,
    mmsi: str,
    date: str,
    segment_id: int,
) -> List[Tuple[int, int]]:
    """
    Find all window indices and their start positions that correspond to
    the given (mmsi, date, segment_id).

    Returns:
        List of (window_idx, start_idx).
    """
    matches: List[Tuple[int, int]] = []

    for win_idx, (seg_idx, start) in enumerate(dataset.windows):
        meta = dataset.segment_meta[seg_idx]
        if (
            meta["mmsi"] == mmsi
            and meta["date"] == date
            and meta["segment_id"] == segment_id
        ):
            matches.append((win_idx, start))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="ais-data/parquet",
        help="Root directory of the AIS parquet dataset.",
    )
    parser.add_argument("--mmsi", type=str, required=True, help="MMSI string (e.g. '209275000').")
    parser.add_argument("--date", type=str, required=True, help="Date string 'YYYY-MM-DD'.")
    parser.add_argument("--segment-id", type=int, required=True, help="Segment id (integer).")
    parser.add_argument(
        "--window-num",
        type=int,
        default=0,
        help=(
            "Rank of the window within this segment (0-based). "
            "For example, 0 = first window, 1 = second, etc."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Window size used during training (default: 30).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help="Minimum raw points in a segment to be considered (default: 30).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # Load checkpoint and model
    # -------------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    in_dim = ckpt["in_dim"]
    hidden_size = ckpt["hidden_size"]
    z_dim = ckpt["z_dim"]
    mean = ckpt["mean"]  # (F,)
    std = ckpt["std"]    # (F,)

    model = VAE_LSTM(
        in_dim=in_dim,
        hidden_size=hidden_size,
        z_dim=z_dim,
        num_layers=1,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------------------------------------------------------------
    # Build dataset and override normalization with checkpoint stats
    # -------------------------------------------------------------
    ds = AISSegmentRaw(
        root_dir=args.root_dir,
        window_size=args.window_size,
        min_points=args.min_points,
        allowed_dates=[args.date],
    )
    ds.mean = mean
    ds.std = std

    # Find windows for the requested segment
    matches = find_windows_for_segment(
        ds,
        mmsi=args.mmsi,
        date=args.date,
        segment_id=args.segment_id,
    )

    if not matches:
        print(
            f"No windows found for MMSI={args.mmsi}, date={args.date}, "
            f"segment_id={args.segment_id}."
        )
        return

    if args.window_num < 0 or args.window_num >= len(matches):
        print(
            f"Invalid window_num={args.window_num}. "
            f"Valid range is [0, {len(matches) - 1}]."
        )
        return

    win_idx, start_idx = matches[args.window_num]
    print(
        f"Using window_num={args.window_num}: "
        f"window_idx={win_idx}, start={start_idx}."
    )

    # -------------------------------------------------------------
    # Get normalized window, run model, and de-normalize
    # -------------------------------------------------------------
    sample = ds[win_idx]["x"]              # (T, F), normalized
    x = sample.unsqueeze(0).to(device)     # (1, T, F)
    B, T, F = x.shape
    mask = torch.ones(B, T, device=device)

    with torch.no_grad():
        x_hat, mu, logvar = model(x, mask)

    # Bring back to CPU and remove batch dimension
    x = x.cpu().squeeze(0)        # (T, F)
    x_hat = x_hat.cpu().squeeze(0)

    # De-normalize using mean and std
    mean = mean.cpu()
    std = std.cpu()
    x_den = x * std + mean        # (T, F)
    x_hat_den = x_hat * std + mean

    # Feature layout (must match dataset_segments_raw):
    # 0: lat
    # 1: lon
    # 2: sog
    # 3: cog_sin
    # 4: cog_cos
    # 5: delta_t
    # 6: ship_type_idx
    # 7: dist_to_cable
    # 8: hour_sin
    # 9: hour_cos
    # 10: dow_sin
    # 11: dow_cos

    lat = x_den[:, 0].numpy()
    lon = x_den[:, 1].numpy()
    sog = x_den[:, 2].numpy()
    cog_sin = x_den[:, 3].numpy()
    cog_cos = x_den[:, 4].numpy()

    lat_hat = x_hat_den[:, 0].numpy()
    lon_hat = x_hat_den[:, 1].numpy()
    sog_hat = x_hat_den[:, 2].numpy()
    cog_sin_hat = x_hat_den[:, 3].numpy()
    cog_cos_hat = x_hat_den[:, 4].numpy()

    # Compute COG in degrees from sin/cos
    def angle_deg(sin_val, cos_val):
        rad = np.arctan2(sin_val, cos_val)
        deg = np.degrees(rad)
        deg = (deg + 360.0) % 360.0
        return deg

    cog = angle_deg(cog_sin, cog_cos)
    cog_hat = angle_deg(cog_sin_hat, cog_cos_hat)

    timesteps = np.arange(T)

    # -------------------------------------------------------------
    # Plot original vs reconstructed
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(timesteps, lat, label="lat (orig)")
    axes[0].plot(timesteps, lat_hat, "--", label="lat (recon)")
    axes[0].set_ylabel("Latitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(timesteps, lon, label="lon (orig)")
    axes[1].plot(timesteps, lon_hat, "--", label="lon (recon)")
    axes[1].set_ylabel("Longitude")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(timesteps, sog, label="SOG (orig)")
    axes[2].plot(timesteps, sog_hat, "--", label="SOG (recon)")
    axes[2].set_ylabel("SOG")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(timesteps, cog, label="COG (orig)")
    axes[3].plot(timesteps, cog_hat, "--", label="COG (recon)")
    axes[3].set_ylabel("COG (deg)")
    axes[3].set_xlabel("Timestep")
    axes[3].legend()
    axes[3].grid(True)

    fig.suptitle(
        f"MMSI={args.mmsi}, Date={args.date}, Segment={args.segment_id}, "
        f"window_num={args.window_num}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = (
        f"recon_example_MMSI{args.mmsi}_D{args.date}"
        f"_S{args.segment_id}_W{args.window_num}.png"
    )
    out_path = Path(out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
