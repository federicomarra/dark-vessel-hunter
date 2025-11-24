import pandas as pd
from typing import List


def add_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    # ensure Timestamp is datetime
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # create Day column so the same Segment number on different days is treated separately
    df['Day'] = df['Timestamp'].dt.date
    # sort including Day to keep chronological order within each day-segment
    df = df.sort_values(["MMSI", "Segment", "Day", "Timestamp"])

    # compute time differences within each (MMSI, Segment, Day) group
    df["DeltaT"] = df.groupby(["MMSI", "Segment", "Day"])["Timestamp"] \
                     .diff().dt.total_seconds().fillna(0)

    # Add unique per-day segment identifier (useful downstream)
    df['Segment_uid'] = df['MMSI'].astype(str) + '_' + df['Segment'].astype(str) + '_' + df['Day'].astype(str)

    df.drop(columns=["Day", "Segment"], inplace=True)

    return df



def split_segments_fixed_length(df: pd.DataFrame, max_len: int = 30) -> pd.DataFrame:
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
    df[df["Segment_nr"] != -1]
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


def one_hot_encode_nav_status(df: pd.DataFrame) -> dict:
    # Create an integer ID for each navigational status value
    df["NavStatusID"] = df["Navigational status"].astype("category").cat.codes

    # save the mapping for future reference (optional but recommended)
    nav_cat = df["Navigational status"].astype("category")
    nav_label_to_id = dict(enumerate(nav_cat.cat.categories))   # id -> label

    # ONE-HOT ENCODING
    nav_dummies = pd.get_dummies(df["NavStatusID"], prefix="NavStatus")
    # concateni al df
    df = pd.concat([df, nav_dummies], axis=1)

    # Dropping original columns
    df = df.drop(columns=["Navigational status", "NavStatusID"])

    return df, nav_label_to_id

def label_ship_types(df: pd.DataFrame) -> dict:
    # Create an integer ID for each ship type
    df["ShipTypeID"] = df["Ship type"].astype("category").cat.codes
    ship_cat = df["Ship type"].astype("category")
    ship_label_to_id = dict(enumerate(ship_cat.cat.categories))  # id -> label
    df.drop(columns=["Ship type"], inplace=True)

    return df, ship_label_to_id