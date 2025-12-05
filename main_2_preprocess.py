# Main preprocess script

# File imports
import config
import src.pre_proc.pre_processing_utils as pre_processing_utils
import src.pre_proc.ais_query as ais_query
import src.pre_proc.ais_segment as ais_segment

# Library imports
from pathlib import Path





def main_preprocess(dataframe_type: str = "all"):
    """
    Orchestrates the preprocessing pipeline for AIS (Automatic Identification System) data.
    This function handles loading raw AIS data from Parquet files, cleaning, feature engineering,
    segmenting tracks, resampling, and saving the processed data for machine learning tasks.
    It supports processing either training or testing datasets based on the provided argument.
    
    The pipeline includes the following steps:
    1.  **Configuration Loading**: Reads parameters (paths, dates, thresholds) from `config.py`.
    2.  **Data Loading**: Queries DuckDB for AIS data within specified date ranges for 'train' or 'test'.
    3.  **Feature Engineering**: Converts COG (Course Over Ground) to sine/cosine components.
    4.  **Cleaning**: Drops unnecessary columns and rows with missing values.
    5.  **Ship Type Grouping**: Aggregates specific ship types into broader categories (Commercial, Passenger, Service, Other).
    6.  **Segmentation**: Splits AIS tracks into segments based on time gaps and duration constraints using `ais_segment`.
    7.  **Filtering**: Removes segments with low point density.
    8.  **Resampling**: Resamples tracks to a fixed time interval.
    9.  **Labeling**: Encodes ship types into numerical IDs.
    10. **Saving**: Exports the processed DataFrame to a Parquet file.
    Args:
        dataframe_type (str, optional): The type of dataset to process.
            - "train": Processes data for the training period defined in config.
            - "test": Processes data for the testing period defined in config.
            - "all": Recursively calls the function for both "train" and "test".
            Defaults to "all".
    Raises:
        ValueError: If `dataframe_type` is not "train", "test", or "all".
    Returns:
        None: The function saves the output to disk (Parquet format) and does not return a value.
    """
    
    # --- 0. HANDLE "all" OPTION ---
    if dataframe_type == "all":
        main_preprocess("train")
        main_preprocess("test")
        return

    # --- 1. CONFIGURATION ---
    # Read configuration from config.py
    VERBOSE_MODE = config.VERBOSE_MODE

    FOLDER_NAME = config.AIS_DATA_FOLDER
    folder_path = Path(FOLDER_NAME)
    parquet_folder_path = folder_path / config.AIS_DATA_FOLDER_PARQUET_SUBFOLDER

    TRAIN_START_DATE = config.TRAIN_START_DATE
    TRAIN_END_DATE = config.TRAIN_END_DATE

    TEST_START_DATE = config.TEST_START_DATE
    TEST_END_DATE = config.TEST_END_DATE

    MAX_TIME_GAP_SEC = config.MAX_TIME_GAP_SEC
    MAX_TRACK_DURATION_SEC = config.MAX_TRACK_DURATION_SEC
    MIN_TRACK_DURATION_SEC = config.MIN_TRACK_DURATION_SEC
    MIN_SEGMENT_LENGTH = config.MIN_SEGMENT_LENGTH

    MIN_FREQ_POINTS_PER_MIN = config.MIN_FREQ_POINTS_PER_MIN

    RESAMPLING_RULE = config.RESAMPLING_RULE
    
    
    # --- 2. LOAD RAW DATAFRAME FROM PARQUET FILE ---        
    if dataframe_type == "train":
        print(f"[preprocess] Querying AIS data for training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
        # Loading filtered data from parquet files
        df = ais_query.query_ais_duckdb(parquet_folder_path, date_start=TRAIN_START_DATE, date_end=TRAIN_END_DATE, verbose=VERBOSE_MODE)
        
    elif dataframe_type == "test":
        print(f"[preprocess] Querying AIS data for testing period: {TEST_START_DATE} to {TEST_END_DATE}")
        # Loading filtered data from parquet files
        df = ais_query.query_ais_duckdb(parquet_folder_path, date_start=TEST_START_DATE, date_end=TEST_END_DATE, verbose=VERBOSE_MODE)
    else:
        raise ValueError(f"Invalid dataframe_type: {dataframe_type}. Must be 'train' or 'test'.")
    
    
    # Converting COG to sine and cosine components
    df = pre_processing_utils.cog_to_sin_cos(df)
    
    # Dropping unnecessary columns and rows with missing values
    df.drop(columns=[ 
        'Type of mobile', 
        'COG', 
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
    
    print("[preprocess] Ship type counts:")
    print(df["Ship type"].value_counts())

    if VERBOSE_MODE:
        print(f"[preprocess] DataFrame after dropping unnecessary columns and NaNs: {len(df):,} rows")

    # Segmenting AIS tracks based on time gaps and max duration, filtering short segments
    df = ais_segment.segment_ais_tracks(
        df,
        max_time_gap_sec=MAX_TIME_GAP_SEC,
        max_track_duration_sec=MAX_TRACK_DURATION_SEC,
        min_track_duration_sec=MIN_TRACK_DURATION_SEC,
        min_track_len=MIN_SEGMENT_LENGTH,
        verbose=VERBOSE_MODE,
        split_long_segments=True,
        discard_leftover=False
    )

    # Adding segment nr feature
    #df = pre_processing_utils.add_segment_nr(df)

    # Removing segments with low point density
    df = pre_processing_utils.remove_notdense_segments(df, min_freq_points_per_min=MIN_FREQ_POINTS_PER_MIN)
    
    # Resampling all tracks to fixed time intervals
    df = pre_processing_utils.resample_all_tracks(df, rule=RESAMPLING_RULE)

    print(f"[preprocess] Number of segments and rows after removing low-density segments and resampling: {df['Segment_nr'].nunique():,} segments, {len(df):,} rows")

    # Ship type labeling (mapping to be used later)
    df, ship_type_to_id = pre_processing_utils.label_ship_types(df)
    
    # Saving pre-processed DataFrame
    if dataframe_type == "train":
        print(f"[preprocess] Saving pre-processed DataFrame to {config.PRE_PROCESSING_DF_TRAIN_PATH}...")
        output_path = config.PRE_PROCESSING_DF_TRAIN_PATH
    else:
        print(f"[preprocess] Saving pre-processed DataFrame to {config.PRE_PROCESSING_DF_TEST_PATH}...")
        output_path = config.PRE_PROCESSING_DF_TEST_PATH

    if VERBOSE_MODE: print(f"[preprocess] Columns of pre-processed DataFrame:\n{df.columns.tolist()}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
        
if __name__ == "__main__":
    main_preprocess()