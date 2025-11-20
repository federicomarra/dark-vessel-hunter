#!/usr/bin/env python3
"""
Compute per-window reconstruction error for a given MMSI / date / segment,
using the trained VAE_LSTM model and the AISSegmentRaw dataset.

Example usage:

    python src/reconstruction_error_segment.py \
        --ckpt runs/best_vae_lstm_chunks.pth \
        --root-dir ais-data/parquet \
        --mmsi 209275000 \
        --date 2025-11-01 \
        --segment-id 0 \
        --save-plot
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt

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
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="If set, save a PNG plot of reconstruction error per window.",
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
    mean = ckpt["mean"]
    std = ckpt["std"]
    mean = mean.to(device)
    std = std.to(device)

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
    # IMPORTANT: allowed_dates should cover at least the date of interest.
    # You can customize this to match your training setup.
    ds = AISSegmentRaw(
        root_dir=args.root_dir,
        window_size=args.window_size,
        min_points=args.min_points,
        allowed_dates=[args.date],  # restrict to this date for speed
    )
    ds.mean = mean
    ds.std = std

    # Find windows that belong to the requested (MMSI, date, segment)
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

    print(
        f"Found {len(matches)} windows for MMSI={args.mmsi}, "
        f"date={args.date}, segment_id={args.segment_id}."
    )

    # -------------------------------------------------------------
    # Compute reconstruction error per window
    # -------------------------------------------------------------
    mse_list = []
    start_list = []

    with torch.no_grad():
        for rank, (win_idx, start_idx) in enumerate(matches):
            sample = ds[win_idx]["x"]  # (T, F), already normalized
            x = sample.unsqueeze(0).to(device)  # (1, T, F)
            B, T, F = x.shape
            mask = torch.ones(B, T, device=device)

            x_hat, mu, logvar = model(x, mask)
            mse = ((x_hat - x) ** 2).mean().item()

            mse_list.append(mse)
            start_list.append(start_idx)

            print(
                f"[{rank}] window_idx={win_idx:5d}, "
                f"start={start_idx:5d}, MSE={mse:.6f}"
            )

    # -------------------------------------------------------------
    # Optional: plot reconstruction error
    # -------------------------------------------------------------
    if args.save_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(mse_list)), mse_list, marker="o")
        ax.set_xlabel("Window index (within this segment)")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(
            f"Reconstruction error\nMMSI={args.mmsi}, Date={args.date}, Segment={args.segment_id}"
        )
        ax.grid(True)

        out_name = f"recon_error_MMSI{args.mmsi}_D{args.date}_S{args.segment_id}.png"
        out_path = Path(out_name)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path.resolve()}")

    print("Done.")


if __name__ == "__main__":
    main()
