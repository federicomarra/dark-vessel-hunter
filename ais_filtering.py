import pandas as pd
import numpy as np

from typing import Sequence, Optional
from shapely.geometry import Point, Polygon
from shapely import contains_xy



def df_filter( df: pd.DataFrame, verbose_mode: bool = False, polygon_filter: bool = True) -> pd.DataFrame:
    """
    Filter AIS dataframe based on bounding box and polygon area.
    Parameters:
    - df: Input AIS dataframe with at least 'Latitude' and 'Longitude' columns
    - verbose_mode: If True, prints filtering progress and statistics
    Returns:
    - Filtered AIS dataframe
    """

    df["MMSI"] = df["MMSI"].astype(str)  # Convert to regular string    
        
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

    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

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
    if polygon_filter:
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

def filter_ais_df(
    df: pd.DataFrame,
    polygon_coords: Sequence[tuple[float, float]],
    allowed_mobile_types: Optional[Sequence[str]] = ("Class A", "Class B"),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply AIS filtering steps to a DataFrame.

    Steps:
    1) Filter by "Type of mobile" (default: keep only "Class A" and "Class B")
    2) MMSI sanity checks (length == 9 and MID in [200, 775])
    3) Drop duplicates on (Timestamp, MMSI)
    4) Polygon filtering using Shapely (lon, lat)

    Parameters
    ----------
    df : pd.DataFrame
        Input AIS DataFrame with at least the columns:
        ["Latitude", "Longitude", "Timestamp", "MMSI"].
    polygon_coords : Sequence[tuple[float, float]]
        Polygon vertices as (lon, lat) pairs.
    allowed_mobile_types : Sequence[str] or None, optional
        Types of mobile to keep (e.g., ["Class A", "Class B"]).
        If None, the "Type of mobile" filter is skipped (if the column exists).
    verbose : bool, optional
        If True, print detailed filtering information.

    Returns
    -------
    pd.DataFrame
        Filtered AIS DataFrame.

    Examples
    --------
    >>> polygon_coords = [
    ...     (10.5162, 57.3500),
    ...     (10.9314, 57.5120),
    ...     (11.5128, 57.5785),
    ...     (11.9132, 57.5230),
    ...     (11.9189, 57.4078),
    ...     (11.2133, 57.1389),
    ...     (11.0067, 57.1352),
    ...     (10.5400, 57.1880),
    ...     (10.5162, 57.3500),
    ... ]
    >>> df_filt = filter_ais_df(df_raw, polygon_coords, verbose=True)
    """
    df = df.copy()

    if verbose:
        print(
            f" [filter_ais_df] Before filtering: {len(df):,} rows, "
            f" [filter_ais_df] {df['MMSI'].nunique():,} unique vessels"
        )

    # ------------------------------------------------------------------
    # 1) Filter by Type of mobile (keep only selected types)
    # ------------------------------------------------------------------
    if "Type of mobile" in df.columns:
        if allowed_mobile_types is not None:
            before_rows = len(df)
            df = df[df["Type of mobile"].isin(allowed_mobile_types)].copy()

            if verbose:
                print(
                    f" [filter_ais_df] Type of mobile filtering complete: {len(df):,} rows "
                    f" [filter_ais_df] (removed {before_rows - len(df):,} rows) "
                    f" [filter_ais_df] using types: {list(allowed_mobile_types)}"
                )
        else:
            if verbose:
                print(
                    " [filter_ais_df] allowed_mobile_types is None, skipping "
                    " [filter_ais_df] 'Type of mobile' filtering step."
                )
    else:
        if verbose:
            print(" [filter_ais_df] Warning: 'Type of mobile' column not found, skipping that filter.")

    # ------------------------------------------------------------------
    # 2) MMSI sanity filters (format + MID)
    # ------------------------------------------------------------------
    # Always start from a clean string
    mmsi_str = df["MMSI"].astype(str).str.strip()

    # Valid MMSI must have 9 digits
    mask_len = mmsi_str.str.len() == 9

    # First 3 digits = MID, must be numeric and in [200, 775]
    mid = mmsi_str.str[:3]
    mask_mid = mid.str.isnumeric() & mid.astype(int).between(200, 775)

    # Combine masks
    valid_mmsi_mask = mask_len & mask_mid

    # Apply once, with aligned index
    df = df[valid_mmsi_mask].copy()

    # Update MMSI column with cleaned values
    df["MMSI"] = mmsi_str[valid_mmsi_mask]

    if verbose:
        print(
            f" [filter_ais_df] MMSI filtering complete: {len(df):,} rows, "
            f" [filter_ais_df] {df['MMSI'].nunique():,} unique vessels"
        )

    # ------------------------------------------------------------------
    # 3) Drop duplicates on (Timestamp, MMSI)
    # ------------------------------------------------------------------
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")

    if verbose:
        print(
            f" [filter_ais_df] Duplicate removal complete: {len(df):,} rows, "
            f" [filter_ais_df] {df['MMSI'].nunique():,} unique vessels"
        )

    # ------------------------------------------------------------------
    # 4) Polygon filtering (FAST, vectorized)
    #    NOTE: Shapely expects (x, y) = (lon, lat)
    # ------------------------------------------------------------------
    polygon = Polygon(polygon_coords)

    lons = df["Longitude"].to_numpy()
    lats = df["Latitude"].to_numpy()

    # Vectorized containment test with Shapely 2.x
    mask_poly = contains_xy(polygon, lons, lats)

    df = df[mask_poly].copy()

    if verbose:
        print(
            f" [filter_ais_df] Polygon filtering complete: {len(df):,} rows, "
            f" [filter_ais_df] {df['MMSI'].nunique():,} unique vessels"
        )

    return df