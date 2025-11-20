"""
Small utility to inspect an AIS segment stored in a Parquet file.

Typical usage:
    python inspect_parquet_segment.py --segment-dir ais-data/parquet/MMSI=209275000/Date=2025-11-01/Segment=0

Or:
    python inspect_parquet_segment.py --file ais-data/parquet/MMSI=209275000/Date=2025-11-01/Segment=0/xxx.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def load_parquet(segment_dir: Path = None, file_path: Path = None) -> pd.DataFrame:
    """Load a parquet file either from a specific file or from a Segment directory."""
    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return pd.read_parquet(file_path)

    if segment_dir is None:
        raise ValueError("You must provide --segment-dir or --file")

    if not segment_dir.exists():
        raise FileNotFoundError(segment_dir)

    # Look for explicit parquet files inside the Segment=x folder
    files = sorted(segment_dir.glob("*.parquet"))
    if not files:
        print(f"[WARN] No .parquet files found inside {segment_dir}. Trying to load directory as dataset...")
        return pd.read_parquet(segment_dir)

    print(f"[INFO] Found {len(files)} parquet files, loading the first one:\n  {files[0]}")
    return pd.read_parquet(files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment-dir",
        type=str,
        default=None,
        help="Path to Segment=x directory (e.g. ais-data/parquet/MMSI=.../Date=.../Segment=0)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Exact parquet file path (overrides --segment-dir)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to print from the top (default: 5)",
    )
    args = parser.parse_args()

    seg_dir = Path(args.segment_dir) if args.segment_dir else None
    file_path = Path(args.file) if args.file else None

    df = load_parquet(segment_dir=seg_dir, file_path=file_path)

    # these rows are useful to see all columns without truncation
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print("\n===== GENERAL INFO =====")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\n===== FIRST ROWS =====")
    print(df.head(args.head))

    # Timestamp analysis
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")
        ts = pd.to_datetime(df["Timestamp"])

        tmin, tmax = ts.min(), ts.max()
        dt_min = (tmax - tmin).total_seconds() / 60
        dt_hr = dt_min / 60

        gaps = ts.diff().dt.total_seconds().fillna(0) / 60  # in minutes

        print("\n===== TIME ANALYSIS =====")
        print("First timestamp :", tmin)
        print("Last timestamp  :", tmax)
        print(f"Total duration  : {dt_min:.2f} minutes ({dt_hr:.2f} hours)")
        print(f"Max gap         : {gaps.max():.2f} minutes")
        print("First 10 gaps (minutes):")
        print(gaps.head(10).to_list())

        # Hour of day / day of week (for circular embedding)
        hour = ts.dt.hour + ts.dt.minute / 60.0
        dow = ts.dt.dayofweek
        print("\n===== HOUR / DAY OF WEEK =====")
        print("Hour of day - min/max:", float(hour.min()), float(hour.max()))
        print("Day of week (0=Monday) - unique values:", sorted(dow.unique()))

    # Basic feature statistics
    def stats(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return "NA"
        return f"min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}"

    print("\n===== BASIC FEATURES: LAT / LON / SOG / COG =====")
    for c in ["Latitude", "Longitude", "SOG", "COG"]:
        if c in df.columns:
            print(f"{c:10s} -> {stats(c)}")
        else:
            print(f"{c:10s} -> NOT PRESENT")

    # Ship type distribution
    if "Ship type" in df.columns:
        print("\n===== SHIP TYPE (top 10 values) =====")
        print(df["Ship type"].value_counts().head(10))

    # Other potentially useful categorical fields
    for c in ["Type of mobile", "Navigational status"]:
        if c in df.columns:
            print(f"\n===== {c.upper()} (top 10 values) =====")
            print(df[c].value_counts().head(10))

    # Distance to cable — check whether it already exists
    print("\n===== DISTANCE TO CABLE =====")
    candidate_cols = [c for c in df.columns if "cable" in c.lower() or "dist" in c.lower()]
    if candidate_cols:
        print("Possible distance columns:", candidate_cols)
        for c in candidate_cols:
            print(f"{c:20s} -> {stats(c)}")
    else:
        print("No column seems to represent 'distance to cable' -> we will compute it ourselves.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
