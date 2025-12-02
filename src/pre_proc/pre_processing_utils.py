import pandas as pd
from typing import List
import numpy as np
import config


def add_segment_nr(df: pd.DataFrame, segment_nr: bool = True) -> pd.DataFrame:
    # ensure Timestamp is datetime
    # create Day column so the same Segment number on different days is treated separately
    df['Day'] = df['Timestamp'].dt.date
    # sort including Day to keep chronological order within each day-segment
    df = df.sort_values(["MMSI", "Segment", "Day", "Timestamp"])

    if segment_nr:
        # Add unique per-day segment identifier (useful downstream)
        df['Segment_nr'] = df['MMSI'].astype(str) + '_' + df['Segment'].astype(str) + '_' + df['Day'].astype(str)

    df.drop(columns=["Day", "Segment"], inplace=True)

    return df

    """
    Takes a DataFrame with a base segment id (Segment_uid) that marks
    continuous tracks of arbitrary length, and splits each track into
    fixed-length segments of `max_len` points.

    - Each new segment gets a unique incremental id in `new_segment_col`
    - Leftover points that do not form a full `max_len` chunk are dropped
    """
    # Ensure deterministic ordering inside each original segment
    df = df.sort_values(["Segment_uid", "Timestamp"])

    # Initialize new segment column
    df["Segment_nr"] = -1  # temporary marker: -1 = "not assigned"

    global_segment_counter = 0

    # Process each continuous track independently
    for seg_uid, group in df.groupby("Segment_uid"):

        n_points = len(group)
        n_full_chunks = n_points // max_len  # integer division

        # For each full chunk of `max_len` points
        for i in range(n_full_chunks):
            global_segment_counter += 1

            start_idx = i * max_len
            end_idx = start_idx + max_len

            # Select rows by POSITION within the group, then map back to df by index
            idx = group.iloc[start_idx:end_idx].index

            # Assign the new segment id
            df.loc[idx, "Segment_nr"] = global_segment_counter

        # Any leftover points (n_points % max_len) are simply ignored
        # because their Segment_nr stays = -1

    # Drop rows that do not belong to a full segment of length max_len
    df = df[df["Segment_nr"] != -1].copy()
    df.drop(columns=["Segment_uid"], inplace=True)

    return df


def normalize_df(df: pd.DataFrame, numeric_cols: List[str]):
    all_values = df[numeric_cols].to_numpy(dtype=float)

    mean = all_values.mean(axis=0)
    std = all_values.std(axis=0)
    # avoid division by zero
    std[std == 0] = 1.0

    df[numeric_cols] = (df[numeric_cols] - mean) / std

    return df, mean, std


def label_ship_types(df: pd.DataFrame) -> dict:
    # Assign IDs according to mapping;
    df["ShipTypeID"] = df["Ship type"].map(lambda x: config.SHIPTYPE_TO_ID.get(x, 2)).astype(int)
    df.drop(columns=["Ship type"], inplace=True)

    return df, config.ID_TO_SHIPTYPE


def cog_to_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two columns 'COG_sin' and 'COG_cos' to the DataFrame,
    converting the COG angle (in degrees) into Cartesian coordinates.
    """
    df = df.copy()
    
    if 'COG' in df.columns:
        # Convert degrees to radians
        radians = np.deg2rad(df['COG'])
        
        # Calculate sine and cosine components
        df['COG_sin'] = np.sin(radians)
        df['COG_cos'] = np.cos(radians)
    else:
        raise ValueError("Column 'COG' not found in DataFrame.")
    
    return df


def easy_resample_interpolate(df_segment: pd.DataFrame, rule: str = '2min') -> pd.DataFrame:
    """
    1. Resample every 'rule' (e.g., '2min').
    2. Lat/Lon/Speed = MEAN of points within the bin.
    3. Linear Interpolation to fill the gaps.
    4. No zero padding (keeps the original data interpolated).
    """
    # 1. Prepare copy and timestamp index
    df = df_segment.copy()
    
    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    df = df.set_index('Timestamp').sort_index()

    # 2. Define WHICH are numeric (Mean) and WHICH are text/static (First value)
    # Add here all numeric columns that should be averaged
    # Note: Circular features (sin/cos) can be averaged safely.
    numeric_cols = ['Latitude', 'Longitude', 'SOG', 'COG_sin', 'COG_cos']
    
    # Filter to keep only those that actually exist in your df
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    # All other columns are "static" (MMSI, Segment, ShipType) -> we take the first value
    static_cols = [c for c in df.columns if c not in numeric_cols]

    # 3. Build the pandas aggregation rules dictionary
    # Example: {'Latitude': 'mean', 'MMSI': 'first', ...}
    agg_rules = {col: 'mean' for col in numeric_cols}
    agg_rules.update({col: 'first' for col in static_cols})

    # 4. RESAMPLE + AGGREGATE (The core logic)
    # Create time bins and apply rules (Mean for numbers, First for static text)
    resampled = df.resample(rule).agg(agg_rules)

    # 5. LINEAR INTERPOLATION (For numeric columns)
    # Fills gaps (NaN) by creating a line between existing points
    resampled[numeric_cols] = resampled[numeric_cols].interpolate(method='linear')

    # 6. Fill static data (MMSI does not interpolate, it drags forward/backward)
    resampled[static_cols] = resampled[static_cols].ffill().bfill()

    # Remove potential rows that remain empty (e.g., if the start of the bin is empty)
    resampled = resampled.dropna(subset=['Latitude', 'Longitude'])

    return resampled.reset_index()


def resample_all_tracks(df: pd.DataFrame, rule: str = '1min') -> pd.DataFrame:
    """
    Applies the 'easy_resample_interpolate' function to every track
    identified by 'Segment_nr' in the DataFrame.
    """
    # 1. Preliminary Timestamp check
    # We convert it once here to avoid doing it inside every group iteration.
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 2. GroupBy + Apply
    # We group by 'Segment_nr' so each track is processed in isolation.
    # include_groups=False avoids FutureWarning; Segment_nr is restored from index later.
    df_resampled = df.groupby("Segment_nr").apply(
        lambda group: easy_resample_interpolate(group, rule=rule),
        include_groups=False
    )
    
    # 3. Final Cleanup
    # Restore Segment_nr from index and ensure the DataFrame index is sequential and clean
    df_resampled = df_resampled.reset_index(level="Segment_nr").reset_index(drop=True)
    
    return df_resampled
