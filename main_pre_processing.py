# Main pre-processing script

# Imports
from pathlib import Path
import pandas as pd
import json

# File imports
import config
import pre_processing_utils
import ais_query

VERBOSE_MODE = config.VERBOSE_MODE

FOLDER_NAME = config.FOLDER_NAME
folder_path = Path(FOLDER_NAME)
parquet_folder_path = folder_path / "parquet"

SEGMENT_MAX_LENGTH = config.SEGMENT_MAX_LENGTH

NUMERIC_COLS = config.NUMERIC_COLS

def main_pre_processing():
    # Loading filtered data from parquet files
    df = ais_query.query_ais_duckdb(parquet_folder_path, verbose=VERBOSE_MODE)

    # Dropping unnecessary columns and rows with missing values
    df.drop(columns=[ 
        'Type of mobile', 
        'ROT', 
        'Heading', 
        'IMO', 
        'Callsign', 
        'Name', 
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


    # Adding △T feature
    df = pre_processing_utils.add_delta_t(df)

    # Splitting segments
    df = pre_processing_utils.split_segments_fixed_length(df, max_len=SEGMENT_MAX_LENGTH)

    # Normalizing numeric columns
    df, mean, std = pre_processing_utils.normalize_df(df, NUMERIC_COLS)

    # Encoding Navicational Status as one-hot
    df, nav_status_to_id = pre_processing_utils.one_hot_encode_nav_status(df)
    NAV_ONEHOT_COLS = [c for c in df.columns if c.startswith("NavStatus_")]

    # Ship type labeling (mapping to be used later)
    df, ship_type_to_id = pre_processing_utils.label_ship_types(df)
    
    # Saving pre-processed DataFrame
    output_path = config.PRE_PROCESSING_DF_PATH
    df.to_parquet(output_path, index=False)

    # Saving preprocessing metadata
    meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "nav_status_to_id": nav_status_to_id,
        "ship_type_to_id": ship_type_to_id
    }

    with open(config.PRE_PROCESSING_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=4)
        
if __name__ == "__main__":
    main_pre_processing()