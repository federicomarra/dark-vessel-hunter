# Module: src/pre_proc/ais_segment.py

# File imports
import config

# Library imports
from typing import Optional
import pandas as pd

def segment_ais_tracks(
    df: pd.DataFrame,
    max_time_gap_sec: Optional[int] = config.MAX_TIME_GAP_SEC,        # e.g. 15 minutes
    max_track_duration_sec: Optional[int] = config.MAX_TRACK_DURATION_SEC,  # e.g. 3 hours
    min_track_duration_sec: Optional[int] = config.MIN_TRACK_DURATION_SEC,  # e.g. 10 minutes
    min_track_len: Optional[int] = config.MIN_SEGMENT_LENGTH,         # e.g. 10 points
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Segment AIS vessel tracks.

    Segmentation is done per MMSI in two stages:

    1) Create segments using:
         - time gaps ≥ max_time_gap_sec
         - segment duration ≥ max_track_duration_sec
       (time gap condition is checked first; if no gap, then duration)
    2) Filter segments using:
         - minimum number of points (min_track_len)
         - minimum segment duration in seconds (min_track_duration_sec)

    Any parameter set to ``None`` disables that rule. If both
    max_time_gap_sec and max_track_duration_sec are None, each MMSI
    gets a single segment (0), and only the min_* filters are applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input AIS data with at least columns: ['MMSI', 'Timestamp', 'SOG'].
    max_time_gap_sec : int or None
        Maximum allowed time gap between consecutive messages within a segment.
        A gap >= this threshold starts a new segment.
    max_track_duration_sec : int or None
        Maximum allowed duration of a segment. If exceeded, a new segment starts.
    min_track_duration_sec : int or None
        Minimum allowed duration of a segment (in seconds) to keep it.
    min_track_len : int or None
        Minimum allowed number of points in a segment to keep it.
    verbose : bool
        If True, prints basic stats before/after filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new integer column 'Segment'.

    """

    df = df.copy()

    # ---------------- Basic checks ----------------
    required_cols = ["MMSI", "Timestamp", "SOG"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"segment_ais_tracks: required columns missing: {missing}")

    df["MMSI"] = df["MMSI"].astype(str)

    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        raise TypeError("segment_ais_tracks: 'Timestamp' must be a datetime dtype")

    if verbose:
        print(
            f"[segment_ais_tracks] Starting with {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} unique vessels"
        )

    # ---------- 1) Sort ---------- 
    df = df.sort_values(["MMSI", "Timestamp"])

    # ---------- 2) Compute Segment IDs (per MMSI) ---------- 
    def compute_segments(ts: pd.Series) -> pd.Series:
        """
        Compute segment IDs for a single MMSI based on:
        - time gaps >= max_time_gap_sec  (checked first)
        - cumulative duration >= max_track_duration_sec

        If both max_time_gap_sec and max_track_duration_sec are None,
        a single segment (0) is returned for all rows.
        """
        ts = ts.sort_values()

        # No segmentation: single segment per MMSI
        if max_time_gap_sec is None and max_track_duration_sec is None:
            return pd.Series(0, index=ts.index, dtype="int64")

        seg_ids = pd.Series(0, index=ts.index, dtype="int64")

        current_seg = 0
        seg_start_time = ts.iloc[0]
        prev_time = ts.iloc[0]

        # First index is already in segment 0
        for idx, t in ts.iloc[1:].items():
            new_segment = False

            # Condition 1: gap-based segmentation (checked first)
            if max_time_gap_sec is not None:
                if (t - prev_time).total_seconds() >= max_time_gap_sec:
                    new_segment = True

            # Condition 2: duration-based segmentation
            if (not new_segment) and (max_track_duration_sec is not None):
                if (t - seg_start_time).total_seconds() >= max_track_duration_sec:
                    new_segment = True

            if new_segment:
                current_seg += 1
                seg_start_time = t  # reset segment start

            seg_ids.at[idx] = current_seg
            prev_time = t

        return seg_ids

    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(compute_segments)

    # ---------- 3) Filter per (MMSI, Segment) ---------- 
    def segment_filter(g: pd.DataFrame) -> bool:
        """
        Returns True if segment passes length and duration constraints.
        Any constraint set to None is ignored.
        """
        # length criterion (>=, not >)
        len_ok = True
        if min_track_len is not None:
            len_ok = len(g) >= min_track_len

        # duration criterion
        time_ok = True
        if min_track_duration_sec is not None:
            dt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds()
            time_ok = dt >= min_track_duration_sec

        return len_ok and time_ok

    if (min_track_len is not None) or (min_track_duration_sec is not None):
        df = df.groupby(["MMSI", "Segment"], group_keys=False).filter(segment_filter)

    df = df.reset_index(drop=True)

    if verbose:
        n_segments = df[["MMSI", "Segment"]].drop_duplicates().shape[0]
        print(
            f"[segment_ais_tracks] After segment-level filter: {len(df):,} rows, "
            f"{n_segments:,} segments"
        )

    return df