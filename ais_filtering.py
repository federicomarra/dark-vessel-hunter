import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon

def df_filter( df: pd.DataFrame, verbose_mode: bool = False) -> pd.DataFrame:
    """
    Filter AIS dataframe based on bounding box and polygon area.
    Parameters:
    - df: Input AIS dataframe with at least 'Latitude' and 'Longitude' columns
    - verbose_mode: If True, prints filtering progress and statistics
    Returns:
    - Filtered AIS dataframe
    """

    # Initial checks (se no ce so queste semo fottuti)
    required_columns = ["Latitude", "Longitude", "# Timestamp", "MMSI", "SOG"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Print initial number of rows and unique vessels
    if verbose_mode:
        print(f"Before filtering: {len(df):,} rows, {df['MMSI'].nunique():,} unique vessels")

    # Bounding box definition (take northest and southest, westest and eastest points)
    bbox = [57.58, 10.5, 57.12, 11.92]  # north lat, west lon, south lat, east lon
    
    # Polygon coordinates definition as (lat, lon) tuples
    polygon_coords = [
        (57.3500, 10.5162),  # coast top left
        (57.5120, 10.9314),  # sea top left
        (57.5785, 11.5128),  # sea top right
        (57.5230, 11.9132),  # top right (Swedish coast)
        (57.4078, 11.9189),  # bottom right (Swedish coast)
        (57.1389, 11.2133),  # sea bottom right
        (57.1352, 11.0067),  # sea bottom left
        (57.1880, 10.5400),  # coast bottom left
        (57.3500, 10.5162),  # close polygon (duplicate of first)
    ]


    # ---- INITIAL FILTERING ----
    df = df.rename(columns={"# Timestamp": "Timestamp"}) # Rename column for consistency
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce") # Convert to datetime

    #df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    #df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first") # Remove duplicates

    # Print how many rows and unique vessels are left after filtering
    if verbose_mode:
        print(f" Initial filtering complete: {len(df):,} rows, {df['MMSI'].nunique():,} unique vessels")


    # ---- BOUNDING BOX FILTERING ----
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (df["Longitude"] <= east)]
    if verbose_mode:
        print(f" Bounding box filtering complete: {len(df):,} rows, {df['MMSI'].nunique():,} unique vessels")


    # ---- POLYGON FILTERING ----
    point = df[["Latitude", "Longitude"]].apply(lambda x: Point(x["Latitude"], x["Longitude"]), axis=1)
    polygon = Polygon(polygon_coords)
    df = df[point.apply(lambda x: polygon.contains(x))]
    if verbose_mode:
        print(f" Polygon filtering complete: {len(df):,} rows, {df['MMSI'].nunique():,} unique vessels")


    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    return df

def split_static_dynamic(df, join_conflicts=True, sep=" | "):
    """
    Split AIS dataframe into static vessel info and dynamic trajectory data.
    Parameters:
    - df: Input AIS dataframe with both static and dynamic columns
    - join_conflicts: If True, joins conflicting static data with separator
    - sep: Separator string used to join conflicting static data
    Returns:
    - static_df: DataFrame with static vessel information
    - dynamic_df: DataFrame with dynamic data
    """
    
    # Define column categories
    STATIC_COLUMNS = [
        'MMSI',
        'IMO',
        'Callsign',
        'Name',
        'Ship type',
        'Cargo type',
        'Width',
        'Length',
        'Size A',
        'Size B',
        'Size C',
        'Size D',
        'Data source type',
        'Type of position fixing device',
    ]
    
    DYNAMIC_COLUMNS = [
        'MMSI',  # Keep as foreign key
        'Timestamp',
        'Type of mobile',
        'Latitude',
        'Longitude',
        'Navigational status',
        'ROT',
        'SOG',
        'COG',
        'Heading',
        'Draught',
        'Destination',
        'ETA',
    ]
    
    if 'MMSI' not in df.columns:
        raise KeyError("MMSI column not found in dataframe")
    
    # 1. CREATE STATIC DATAFRAME
    available_static = [col for col in STATIC_COLUMNS if col in df.columns]
    agg_cols = [col for col in available_static if col != 'MMSI']
    
    def _agg(series):
        vals = series.dropna().unique().tolist()
        if len(vals) == 0:
            return np.nan
        if len(vals) == 1:
            return vals[0]
        if join_conflicts:
            if "Unknown" in vals:
                vals.remove("Unknown")
            if "Undefined" in vals:
                vals.remove("Undefined")
            if len(vals) == 1:
                return vals[0]
            return sep.join(map(str, vals))
        return vals
    
    static_df = df.groupby('MMSI')[agg_cols].agg(_agg).reset_index()
    
    
    # 2. CREATE DYNAMIC DATAFRAME
    available_dynamic = [col for col in DYNAMIC_COLUMNS if col in df.columns]
    dynamic_df = df[available_dynamic].copy()
    
    # 3. REPORT
    print(f"Split complete:")
    print(f"   Static:  {len(static_df):,} unique vessels with {len(static_df.columns)} columns")
    print(f"   Dynamic: {len(dynamic_df):,} AIS messages with {len(dynamic_df.columns)} columns")
    
    # Check for conflicts in static data
    conflict_cols = []
    for col in agg_cols:
        if static_df[col].astype(str).str.contains(sep, regex=False).any():
            n_conflicts = static_df[col].astype(str).str.contains(sep, regex=False).sum()
            conflict_cols.append(f"{col} ({n_conflicts})")
    
    if conflict_cols:
        print(f"  Static conflicts: {', '.join(conflict_cols)}")
    
    return static_df, dynamic_df