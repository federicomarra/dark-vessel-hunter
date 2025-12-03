# Main pre-processing script

# File imports
import config
import src.pre_proc.pre_processing_utils as pre_processing_utils
import src.pre_proc.ais_query as ais_query
import src.pre_proc.ais_segment as ais_segment

# Library imports
from pathlib import Path
import pandas as pd
import json

VERBOSE_MODE = config.VERBOSE_MODE

FOLDER_NAME = config.AIS_DATA_FOLDER
folder_path = Path(FOLDER_NAME)
parquet_folder_path = folder_path / "parquet"

MAX_TIME_GAP_SEC = config.MAX_TIME_GAP_SEC
MAX_TRACK_DURATION_SEC = config.MAX_TRACK_DURATION_SEC
MIN_TRACK_DURATION_SEC = config.MIN_TRACK_DURATION_SEC
MIN_SEGMENT_LENGTH = config.MIN_SEGMENT_LENGTH

MIN_FREQ_POINTS_PER_MIN = config.MIN_FREQ_POINTS_PER_MIN


NUMERIC_COLS = config.NUMERIC_COLS
# if u want to do it withouth a end date comment next line
TRAIN_START_DATE = config.TRAIN_START_DATE
TRAIN_END_DATE = config.TRAIN_END_DATE

TEST_START_DATE = config.TEST_START_DATE
TEST_END_DATE = config.TEST_END_DATE

RESAMPLING_RULE = config.RESAMPLING_RULE

def main_pre_processing(dataframe_type: str = "all"):

    if dataframe_type == "all":
        main_pre_processing("train")
        main_pre_processing("test")
        return
        
    elif dataframe_type == "train":
        print(f"[pre_processing] Querying AIS data for training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
        # Loading filtered data from parquet files
        dates = (
            pd.date_range(TRAIN_START_DATE, TRAIN_END_DATE, freq="D")
            .strftime("%Y-%m-%d")
            .tolist()
        )
        df = ais_query.query_ais_duckdb(parquet_folder_path, dates=dates, verbose=VERBOSE_MODE)
        
    elif dataframe_type == "test":
        print(f"[pre_processing] Querying AIS data for testing period: {TEST_START_DATE} to {TEST_END_DATE}")
        # Loading filtered data from parquet files
        dates = (
            pd.date_range(TEST_START_DATE, TEST_END_DATE, freq="D")
            .strftime("%Y-%m-%d")
            .tolist()
        )
        df = ais_query.query_ais_duckdb(parquet_folder_path, dates=dates, verbose=VERBOSE_MODE)
    else:
        raise ValueError(f"Invalid dataframe_type: {dataframe_type}. Must be 'train' or 'test'.")
    
    # Converting COG to sine and cosine components
    df = pre_processing_utils.cog_to_sin_cos(df)
    
    # Dropping unnecessary columns and rows with missing values
    df.drop(columns=[ 
        'Type of mobile', 
        'ROT',
        'COG',
        'Heading', 
        'IMO', 
        'Callsign', 
        'Name', 
        'Navigational status',
        'Cargo type', 
        'Width', 
        'Length',
        'Type of position fixing device', 
        'Draught', 
        'Destination', 
        'ETA',
        'Data source type', 
        'A', 'B', 'C', 'D', 
        'Date'], inplace=True, errors='ignore')
    
    # Removing rows with NaN values in essential columns
    df.dropna(inplace=True)
    
    # Grouping Ship types
    commercial_types = ["Cargo", "Tanker"]
    passenger_types = ["Passenger", "Pleasure", "Sailing"]
    service_types = ["Dredging", "Law enforcement", "Military", "Port tender", "SAR", "Towing", "Towing long/wide","Tug"]
    valid_types =  ["Fishing", "Service", "Commercial", "Passenger"]

    df.loc[df["Ship type"].isin(commercial_types), "Ship type"] = "Commercial"
    df.loc[df["Ship type"].isin(passenger_types), "Ship type"] = "Passenger"
    df.loc[df["Ship type"].isin(service_types), "Ship type"] = "Service"
    df.loc[~df["Ship type"].isin(valid_types), "Ship type"] = "Other"
    
    if VERBOSE_MODE:
        print(f"[pre_processing] DataFrame after dropping unnecessary columns and NaNs: {len(df):,} rows")

    print("[pre_processing] Number of tracks before segmentation:", df["TrackID"].nunique())
    # Segmenting AIS tracks based on time gaps and max duration, filtering short segments
    df = ais_segment.segment_ais_tracks(
        df,
        max_time_gap_sec=MAX_TIME_GAP_SEC,
        max_track_duration_sec=MAX_TRACK_DURATION_SEC,
        min_track_duration_sec=MIN_TRACK_DURATION_SEC,
        min_track_len=MIN_SEGMENT_LENGTH,
        verbose=VERBOSE_MODE
    )

    print("[pre_processing] Ship type counts:")
    print(df["Ship type"].value_counts())

    # Adding segment uid feature
    df = pre_processing_utils.add_segment_nr(df)
    print(f"[pre_processing] Number of segments after segmentation: {df['Segment_nr'].nunique():,}")

    # Removing segments with low point density
    df = pre_processing_utils.remove_notdense_segments(df, min_freq_points_per_min=MIN_FREQ_POINTS_PER_MIN)
    print(f"[pre_processing] Number of segments after removing low-density segments: {df['Segment_nr'].nunique():,}")
    
    # Resampling all tracks to fixed time intervals
    df = pre_processing_utils.resample_all_tracks(df, rule=RESAMPLING_RULE)

    # Normalizing numeric columns
    df, mean, std = pre_processing_utils.normalize_df(df, NUMERIC_COLS)

    # Ship type labeling (mapping to be used later)
    df, ship_type_to_id = pre_processing_utils.label_ship_types(df)
    
    # Saving pre-processed DataFrame
    if dataframe_type == "train":
        print(f"[pre_processing] Saving pre-processed DataFrame to {config.PRE_PROCESSING_DF_TRAIN_PATH}...")
        output_path = config.PRE_PROCESSING_DF_TRAIN_PATH
        metadata_path = config.PRE_PROCESSING_METADATA_TRAIN_PATH
    else:
        print(f"[pre_processing] Saving pre-processed DataFrame to {config.PRE_PROCESSING_DF_TEST_PATH}...")
        output_path = config.PRE_PROCESSING_DF_TEST_PATH
        metadata_path = config.PRE_PROCESSING_METADATA_TEST_PATH

    if VERBOSE_MODE: print(f"[pre_processing] Columns of pre-processed DataFrame:\n{df.columns.tolist()}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Saving preprocessing metadata
    if VERBOSE_MODE: print(f"[pre_processing] Saving preprocessing metadata to {metadata_path}...")
    meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ship_type_to_id": ship_type_to_id
    }

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=4)
        
if __name__ == "__main__":
    main_pre_processing()