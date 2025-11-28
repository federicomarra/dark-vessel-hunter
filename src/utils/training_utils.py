import os
import pandas as pd
import config

NUMERIC_COLS = config.NUMERIC_COLS
FEATURE_COLS = NUMERIC_COLS


def load_df_seq(data_path: str) -> pd.DataFrame:
    """
    Load df_seq from a file.

    Expected columns:
      - 'Segment_nr'
      - 'MMSI'
      - 'ShipTypeID'
      - 'Sequence' : list-of-lists or np.ndarray with shape (T, F)

    You can adapt this function to match how you store df_seq.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    sequences = []

    # ----- Group by Segment_nr to create sequences ------
    groups = df.groupby("Segment_nr")
    for seg_id, g in groups:
        g = g.sort_values("Timestamp")
        if len(g) != config.SEGMENT_MAX_LENGTH:
            continue  # safety: skip anomalous segments

        X = g[FEATURE_COLS].to_numpy(dtype=float)
        ship_type_id = int(g["ShipTypeID"].iloc[0])
        mmsi = g["MMSI"].iloc[0]
        first_timestamp = g["Timestamp"].iloc[0]

        sequences.append({
            "Segment_nr": seg_id,
            "MMSI": mmsi,
            "ShipTypeID": ship_type_id,
            "FirstTimestamp": first_timestamp,
            "Sequence": X,
        })

    df_seq = pd.DataFrame(sequences)
       
    return df_seq