# Main pre-processing script

# File imports
import config
import src.pre_proc.pre_processing_utils as pre_processing_utils
import src.pre_proc.ais_query as ais_query

# Library imports
from pathlib import Path
import pandas as pd
import json

VERBOSE_MODE = config.VERBOSE_MODE

FOLDER_NAME = config.AIS_DATA_FOLDER
folder_path = Path(FOLDER_NAME)
parquet_folder_path = folder_path / "parquet"

SEGMENT_MAX_LENGTH = config.SEGMENT_MAX_LENGTH

NUMERIC_COLS = config.NUMERIC_COLS
# if u want to do it withouth a end date comment next line
TRAIN_START_DATE = config.TRAIN_START_DATE
TRAIN_END_DATE = config.TRAIN_END_DATE

TEST_START_DATE = config.TEST_START_DATE
TEST_END_DATE = config.TEST_END_DATE

def main_pre_processing(dataframe_type: str = "train"):

    if dataframe_type == "train":
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
    

    # Dropping unnecessary columns and rows with missing values
    print(f"[pre_processing] Initial data size: {len(df)} records.")
    print(f"[pre_processing] Dropping unnecessary columns and rows with missing values...")
    df.drop(columns=[ 
        'Type of mobile', 
        'ROT', 
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

    df.dropna(inplace=True)
    print(f"[pre_processing] Data size after dropping: {len(df)} records.")


    # Adding △T feature
    df = pre_processing_utils.add_delta_t(df)
    df.drop(columns=["DeltaT"], inplace=True)

    # Splitting segments
    print(f"[pre_processing] Splitting segments to max length {SEGMENT_MAX_LENGTH}...")
    df = pre_processing_utils.split_segments_fixed_length(df, max_len=SEGMENT_MAX_LENGTH)

    # Normalizing numeric columns
    df, mean, std = pre_processing_utils.normalize_df(df, NUMERIC_COLS)

    # Encoding Navicational Status as one-hot
    #df, nav_status_to_id = pre_processing_utils.one_hot_encode_nav_status(df)

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

    print(f"[pre_processing] Columns of pre-processed DataFrame:\n{df.columns.tolist()}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Saving preprocessing metadata
    print(f"[pre_processing] Saving preprocessing metadata to {metadata_path}...")
    meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        #"nav_status_to_id": nav_status_to_id,
        "ship_type_to_id": ship_type_to_id
    }

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=4)
        
if __name__ == "__main__":
    main_pre_processing()