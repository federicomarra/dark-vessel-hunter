# Module: src/pre_proc/ais_segment.py

# File imports
import config

# Library imports
from typing import Optional
import pandas as pd
import numpy as np

def segment_ais_tracks(
    df: pd.DataFrame,
    max_time_gap_sec: Optional[int] = config.MAX_TIME_GAP_SEC,              # e.g. 15 minutes
    max_track_duration_sec: Optional[int] = config.MAX_TRACK_DURATION_SEC,  # e.g. 12 hours
    min_track_duration_sec: Optional[int] = config.MIN_TRACK_DURATION_SEC,  # e.g. 15 minutes
    min_track_len: Optional[int] = config.MIN_SEGMENT_LENGTH,               # e.g. 10 points
    discard_leftover: bool = False,                                         # Flag to discard excess duration (only if splitting)
    split_long_segments: bool = True,                                      # New Flag: Toggle between Cutting vs Filtering
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Segment AIS vessel tracks using a two-pass approach.

    Logic:
    1. 'Natural' Segmentation: Split tracks based on time gaps (max_time_gap_sec) 
       and vessel ID changes.
    2. First Filter: Discard natural segments that are too short/small.
    3. Duration Handling (Max Duration):
       - If split_long_segments=True (Default): Cut segments > max_duration into chunks.
         (Uses discard_leftover to decide if we keep the chunk after the cut).
       - If split_long_segments=False: Strictly discard any segment > max_duration.
    4. Final Filter: Discard any resulting segments (including leftovers) that are 
       now too short.

    Parameters
    ----------
    df : pd.DataFrame
        Input AIS data (must have MMSI, Timestamp).
    max_time_gap_sec : int, optional
        Max gap allowed before starting a new segment.
    max_track_duration_sec : int, optional
        Max duration allowed for a single segment.
    min_track_duration_sec : int, optional
        Minimum duration required to keep a segment.
    min_track_len : int, optional
        Minimum number of points required to keep a segment.
    discard_leftover : bool
        Only used if split_long_segments=True.
        If True, data exceeding max_track_duration_sec is deleted. 
        If False, it becomes a new segment.
    split_long_segments : bool
        If True, segments longer than max_track_duration_sec are cut into smaller chunks.
        If False, segments longer than max_track_duration_sec are discarded entirely.
    verbose : bool
        Print stats.

    Returns
    -------
    pd.DataFrame
        DataFrame with column 'Segment_nr'.
    """

    df = df.copy()

    # ---------------- Basic checks ----------------
    required_cols = ["MMSI", "Timestamp"]
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

    # 1) Sort
    df = df.sort_values(["MMSI", "Timestamp"])

    # ==============================================================================
    # PHASE 1: Natural Segmentation (Time Gaps Only)
    # ==============================================================================
    
    # Calculate time diff between points per vessel
    # fillna(0.0) handles the very first point of the dataframe safely
    df['dt_gap'] = df.groupby('MMSI')['Timestamp'].diff().dt.total_seconds().fillna(0.0)
    
    # Determine split conditions
    # 1. Vessel changed
    # 2. Time gap is too large (if parameter is set)
    cond_mmsi_change = (df['MMSI'] != df['MMSI'].shift())
    
    if max_time_gap_sec is not None:
        cond_gap = (df['dt_gap'] >= max_time_gap_sec)
        is_new_seg = cond_mmsi_change | cond_gap
    else:
        is_new_seg = cond_mmsi_change
    
    # Create temporary natural IDs
    df['temp_seg_id'] = is_new_seg.cumsum()

    # ==============================================================================
    # PHASE 2: First Filter (Discard "Natural" Short Segments)
    # ==============================================================================
    # We remove natural segments that are already too short before spending time cutting them.
    
    if (min_track_duration_sec is not None) or (min_track_len is not None):
        
        # Optimized aggregation: Get min/max/count in one go.
        seg_stats = df.groupby('temp_seg_id')['Timestamp'].agg(['min', 'max', 'count'])
        
        # Calculate duration in seconds (Vectorized)
        durations = (seg_stats['max'] - seg_stats['min']).dt.total_seconds()
        
        # Build Filter Mask
        keep_mask = pd.Series(True, index=seg_stats.index)
        
        if min_track_duration_sec is not None:
            keep_mask &= (durations >= min_track_duration_sec)
            
        if min_track_len is not None:
            keep_mask &= (seg_stats['count'] >= min_track_len)
            
        # Apply Filter
        valid_ids = seg_stats.index[keep_mask]
        df = df[df['temp_seg_id'].isin(valid_ids)].copy()

    # ==============================================================================
    # PHASE 3: Handle Long Segments (Cut vs Filter)
    # ==============================================================================
    
    if max_track_duration_sec is not None:
        
        if split_long_segments:
            # --- OPTION A: CUTTING LOGIC ---
            # Cut segments into chunks of max_track_duration_sec
            
            # Calculate elapsed time relative to the start of the *current natural segment*
            seg_start_times = df.groupby('temp_seg_id')['Timestamp'].transform('first')
            df['elapsed_from_start'] = (df['Timestamp'] - seg_start_times).dt.total_seconds()

            # "Chunking": Divide elapsed time by max duration.
            df['chunk_id'] = (df['elapsed_from_start'] // max_track_duration_sec).astype(int)

            if discard_leftover:
                # Keep only the first chunk (0). Drop everything else.
                df = df[df['chunk_id'] == 0].copy()
                df['final_grouping_key'] = df['temp_seg_id']
            else:
                # Keep leftovers as new segments.
                # Group by (temp_seg_id, chunk_id)
                df['final_grouping_key'] = list(zip(df['temp_seg_id'], df['chunk_id']))
                
        else:
            # --- OPTION B: FILTERING LOGIC ---
            # Strictly discard segments that are longer than max_track_duration_sec
            
            # We aggregate again to check the current durations
            seg_stats_max = df.groupby('temp_seg_id')['Timestamp'].agg(['min', 'max'])
            durations_max = (seg_stats_max['max'] - seg_stats_max['min']).dt.total_seconds()
            
            # Find IDs that are WITHIN the limit
            valid_max_ids = seg_stats_max.index[durations_max <= max_track_duration_sec]
            
            df = df[df['temp_seg_id'].isin(valid_max_ids)].copy()
            df['final_grouping_key'] = df['temp_seg_id']
            
    else:
        # No max duration limit set
        df['final_grouping_key'] = df['temp_seg_id']

    # Generate the requested "Segment_nr"
    # ngroup() assigns a unique integer 0..N to each unique grouping key
    df['Segment_nr'] = df.groupby('final_grouping_key', sort=False).ngroup()

    # ==============================================================================
    # PHASE 4: Final Filter (Clean up Leftovers)
    # ==============================================================================
    # This phase is crucial if we did splitting (Option A).
    # If we did Option B, this is mostly redundant but harmless (safety check).
    
    if (min_track_duration_sec is not None) or (min_track_len is not None):
        
        seg_stats = df.groupby('Segment_nr')['Timestamp'].agg(['min', 'max', 'count'])
        durations = (seg_stats['max'] - seg_stats['min']).dt.total_seconds()
        
        keep_mask = pd.Series(True, index=seg_stats.index)
        
        if min_track_duration_sec is not None:
            keep_mask &= (durations >= min_track_duration_sec)
            
        if min_track_len is not None:
            keep_mask &= (seg_stats['count'] >= min_track_len)
            
        valid_ids = seg_stats.index[keep_mask]
        df = df[df['Segment_nr'].isin(valid_ids)].copy()
        
        # Optional: Re-number segments to be sequential after dropping some
        df['Segment_nr'] = df.groupby('Segment_nr', sort=False).ngroup()

    # Cleanup temporary columns
    drop_cols = ['dt_gap', 'temp_seg_id', 'elapsed_from_start', 'chunk_id', 'final_grouping_key']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).reset_index(drop=True)

    if verbose:
        n_segments = df['Segment_nr'].nunique() if not df.empty else 0
        print(
            f"[segment_ais_tracks] Final processed data: {len(df):,} rows, "
            f"{n_segments:,} segments."
        )

    return df