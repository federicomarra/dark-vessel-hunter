from pathlib import Path
from typing import Optional, Union
import shutil

import pandas as pd
import pyarrow
import pyarrow.parquet


def segment_ais_tracks(
    df: pd.DataFrame,
    min_track_len: int = 256,
    min_track_duration_sec: int = 60 * 60,
    max_time_gap_sec: int = 15 * 60,
    sog_min: Optional[float] = 0.5,
    sog_max: Optional[float] = 25.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter AIS tracks and segment them by time gaps.

    Assumptions
    -----------
    - `Timestamp` is already a datetime64 dtype.
    - `SOG` is in **m/s**, and `sog_min` / `sog_max` (if given) are in m/s.

    Steps
    -----
    1) Per-MMSI track filtering:
       - length > min_track_len
       - duration >= min_track_duration_sec
       - optional SOG range [sog_min, sog_max]
    2) Sort by (MMSI, Timestamp)
    3) Define `Segment` via time gaps > `max_time_gap_sec`
    4) Apply the same filter at (MMSI, Segment) level
    5) Add `Date` column: YYYY-MM-DD

    Parameters
    ----------
    df : pd.DataFrame
        Filtered AIS DataFrame containing at least:
        ["MMSI", "Timestamp", "SOG"].
    min_track_len : int, optional
        Minimum number of points required for track/segment.
    min_track_duration_sec : int, optional
        Minimum duration in seconds for track/segment.
    max_time_gap_sec : int, optional
        Maximum allowed time gap in seconds within a segment.
    sog_min : float or None, optional
        Minimum SOG (m/s) for valid track/segment. If None, no lower bound.
    sog_max : float or None, optional
        Maximum SOG (m/s) for valid track/segment. If None, no upper bound.
    verbose : bool, optional
        If True, print information about filtering and segmentation.

    Returns
    -------
    pd.DataFrame
        DataFrame with valid tracks, including:
        - "MMSI"
        - "Timestamp"
        - "SOG"
        - "Segment" (int)
        - "Date" (str, YYYY-MM-DD)

    Examples
    --------
    >>> df_seg = segment_ais_tracks(df_filt, verbose=True)
    >>> df_seg[['MMSI', 'Timestamp', 'Segment']].head()
    """
    df = df.copy()

    required_cols = ["MMSI", "Timestamp", "SOG"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f" segment_ais_tracks: required columns missing: {missing}")

    df["MMSI"] = df["MMSI"].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        raise TypeError(" segment_ais_tracks: 'Timestamp' must be a datetime dtype")

    if verbose:
        print(
            f" [segment_ais_tracks] Starting with {len(df):,} rows, "
            f" { df['MMSI'].nunique():,} unique vessels"
        )

    # helper: track filter 
    def track_filter(g: pd.DataFrame) -> bool:
        len_ok = len(g) > min_track_len

        if sog_min is not None or sog_max is not None:
            sog_max_val = g["SOG"].max()
            sog_ok = True
            if sog_min is not None:
                sog_ok &= sog_max_val >= sog_min
            if sog_max is not None:
                sog_ok &= sog_max_val <= sog_max
        else:
            sog_ok = True

        dt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds()
        time_ok = dt >= min_track_duration_sec

        return len_ok and sog_ok and time_ok

    # ---------- 1) Filter per MMSI ----------
    df = df.groupby("MMSI", group_keys=False).filter(track_filter)

    if verbose:
        print(
            f" [segment_ais_tracks] After MMSI-level filter: {len(df):,} rows, "
            f" {df['MMSI'].nunique():,} vessels"
        )

    # ---------- 2) Sort ----------
    df = df.sort_values(["MMSI", "Timestamp"])

    # ---------- 3) Compute Segment IDs ----------
    def compute_segments(ts: pd.Series) -> pd.Series:
        gaps = ts.diff().dt.total_seconds().fillna(0)
        return (gaps >= max_time_gap_sec).cumsum()

    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(compute_segments)

    # ---------- 4) Filter per (MMSI, Segment) ----------
    df = df.groupby(["MMSI", "Segment"], group_keys=False).filter(track_filter)
    df = df.reset_index(drop=True)

    if verbose:
        print(
            f" [segment_ais_tracks] After segment-level filter: {len(df):,} rows, "
            f" {df[['MMSI','Segment']].drop_duplicates().shape[0]:,} segments"
        )

    # ---------- 5) Add Date ----------
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    return df


def save_by_mmsi(
    df: pd.DataFrame,
    verbose: bool = False,
    output_folder: Union[Path, str] = "ais_data_parquet",
) -> Path:
    """
    Write AIS data to a partitioned Parquet dataset.

    The output directory is **always** "ais_data_parquet".
    If it does not exist, it will be created.

    IMPORTANT
    ---------
    This function is *overwrite-safe* for the partitions present in `df`.
    For each unique (MMSI, Date, Segment) combination in `df`, the existing
    partition directory is removed before writing new data. This avoids
    accumulating multiple parquet files for the same segment when rerunning
    the pipeline for the same date/file.

    Expected columns
    ----------------
    df must contain:
    - "MMSI"    (string-like)
    - "Date"    (string, e.g. "2025-11-05")
    - "Segment" (int)

    Partition layout
    ----------------
    ais_data_parquet/
        MMSI=123456789/
            Date=2025-11-05/
                Segment=0/part-*.parquet
                Segment=1/part-*.parquet
        MMSI=987654321/
            Date=2025-11-06/
                Segment=0/part-*.parquet

    Parameters
    ----------
    df : pd.DataFrame
        Segmented AIS DataFrame containing "MMSI", "Date", "Segment".
    verbose : bool, optional
        If True, print the output path and some info.

    Returns
    -------
    Path
        Path to the root parquet dataset folder ("ais_data_parquet").

    Examples
    --------
    >>> df_seg = segment_ais_tracks(df_filt)
    >>> out_root = save_by_mmsi(df_seg, verbose=True)
    """
    df = df.copy()

    required_cols = ["MMSI", "Date", "Segment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f" save_by_mmsi: required columns missing: {missing}")

    df["MMSI"] = df["MMSI"].astype(str)
    
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Remove existing segment folders for (MMSI, Date, Segment) in df
    # ------------------------------------------------------------------
    partitions = df[["MMSI", "Date", "Segment"]].drop_duplicates()

    for _, row in partitions.iterrows():
        mmsi_val = row["MMSI"]
        date_val = row["Date"]
        seg_val = row["Segment"]

        seg_dir = (
            out_path
            / f"MMSI={mmsi_val}"
            / f"Date={date_val}"
            / f"Segment={seg_val}"
        )

        if seg_dir.exists():
            if verbose:
                print(f" [save_by_mmsi] Removing existing partition: {seg_dir}")
            shutil.rmtree(seg_dir)

    # ------------------------------------------------------------------
    # Write new dataset (append is fine now that partitions are cleaned)
    # ------------------------------------------------------------------
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=["MMSI", "Date", "Segment"],
    )

    if verbose:
        print(f" [save_by_mmsi] Parquet dataset written/appended at: {out_path.resolve()}")

    return out_path
