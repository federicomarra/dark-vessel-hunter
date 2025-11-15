import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

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

    # ---- REMOVE SHIPS WITH SOG = 0 FOR MORE THAN 90% OF THEIR DATA ----
    sog_zero_threshold = 0.9  # 90%
    sog_zero_stats = df.groupby("MMSI")["SOG"].apply(lambda x: (x <= 0).mean())
    mmsi_to_remove = sog_zero_stats[sog_zero_stats > sog_zero_threshold].index
    df = df[~df["MMSI"].isin(mmsi_to_remove)]
    if verbose_mode:
        print(f" Removed ships with >90% SOG = 0: {len(mmsi_to_remove):,} vessels")

        # ---- MIN MOVEMENT FILTERING (LAT/LON RANGE) ----
    # keep only vessels that move at least 0.01° in lat OR lon
    movement_stats = df.groupby("MMSI").agg(
        lat_min=("Latitude", "min"),
        lat_max=("Latitude", "max"),
        lon_min=("Longitude", "min"),
        lon_max=("Longitude", "max"),
    )

    lat_range = movement_stats["lat_max"] - movement_stats["lat_min"]
    lon_range = movement_stats["lon_max"] - movement_stats["lon_min"]

    movement_mask = (lat_range >= 0.01) | (lon_range >= 0.01)

    mmsi_keep = movement_stats.index[movement_mask]
    mmsi_removed_movement = movement_stats.index[~movement_mask]

    df = df[df["MMSI"].isin(mmsi_keep)]

    if verbose_mode:
        print(f" Removed low-movement vessels (<0.01° lat & lon): {len(mmsi_removed_movement):,} vessels")
        print(f" Final dataframe: {len(df):,} rows, {df['MMSI'].nunique():,} unique vessels")


    return df


def split_static_dynamic(df, join_conflicts=True, verbose_mode=False, sep=" | "):
    """
    Split AIS dataframe into static vessel info and dynamic trajectory data.
    Adds temporal statistics to static dataframe.
    
    Parameters:
    - df: Input AIS dataframe with both static and dynamic columns
    - join_conflicts: If True, joins conflicting static data with separator
    - sep: Separator string used to join conflicting static data
    
    Returns:
    - static_df: DataFrame with static vessel information + temporal stats
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
    
    # Ensure Timestamp is datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # ==========================================
    # 1. CREATE STATIC DATAFRAME
    # ==========================================
    available_static = [col for col in STATIC_COLUMNS if col in df.columns]
    agg_cols = [col for col in available_static if col != 'MMSI']
    
    def _agg(series):
        vals = series.dropna().unique().tolist()
        if len(vals) == 0:
            return np.nan
        if len(vals) == 1:
            return vals[0]
        if join_conflicts:
            # Remove common "missing" values
            vals = [v for v in vals if str(v).lower() not in ['unknown', 'undefined', 'nan', 'none']]
            if len(vals) == 0:
                return np.nan
            if len(vals) == 1:
                return vals[0]
            return sep.join(map(str, vals))
        return vals
    
    static_df = df.groupby('MMSI')[agg_cols].agg(_agg).reset_index()
    
    # ==========================================
    # 2. ADD TEMPORAL STATISTICS TO STATIC DF
    # ==========================================
    if verbose_mode:
        print("📊 Calculating temporal statistics...")
    
    temporal_stats = []
    
    for mmsi in static_df['MMSI']:
        vessel_data = df[df['MMSI'] == mmsi]
        
        stats = {'MMSI': mmsi}
        
        # Message count
        stats['message_count'] = len(vessel_data)
        
        # Temporal information
        if 'Timestamp' in df.columns:
            valid_timestamps = vessel_data['Timestamp'].dropna()
            
            if len(valid_timestamps) > 0:
                stats['first_seen'] = valid_timestamps.min()
                stats['last_seen'] = valid_timestamps.max()
                
                # Time in area (hours)
                time_span = (stats['last_seen'] - stats['first_seen']).total_seconds() / 3600
                stats['time_in_area_hours'] = round(time_span, 2)
                
                # Messages per hour (avoid division by zero)
                if time_span > 0:
                    stats['avg_messages_per_hour'] = round(len(vessel_data) / time_span, 2)
                else:
                    stats['avg_messages_per_hour'] = len(vessel_data)
                
                # Unique days
                stats['unique_days'] = vessel_data['Timestamp'].dt.date.nunique()
                
                # Activity pattern (transit, fishing, anchored, etc.)
                if time_span < 2:  # Less than 2 hours
                    stats['activity_pattern'] = 'transit (<2h)'
                elif time_span < 24:  # Less than 1 day
                    stats['activity_pattern'] = 'short_stay (<24h)'
                elif time_span < 168:  # Less than 1 week
                    stats['activity_pattern'] = 'medium_stay (<7d)'
                else:
                    stats['activity_pattern'] = 'long_stay'
            else:
                stats['first_seen'] = pd.NaT
                stats['last_seen'] = pd.NaT
                stats['time_in_area_hours'] = 0
                stats['avg_messages_per_hour'] = 0
                stats['unique_days'] = 0
                stats['activity_pattern'] = 'unknown'
        
        # Geographic statistics
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            stats['lat_min'] = vessel_data['Latitude'].min()
            stats['lat_max'] = vessel_data['Latitude'].max()
            stats['lon_min'] = vessel_data['Longitude'].min()
            stats['lon_max'] = vessel_data['Longitude'].max()
            stats['lat_center'] = vessel_data['Latitude'].mean()
            stats['lon_center'] = vessel_data['Longitude'].mean()
            
            # Geographic spread (rough estimate in degrees)
            lat_range = stats['lat_max'] - stats['lat_min']
            lon_range = stats['lon_max'] - stats['lon_min']
            stats['geographic_spread_deg'] = round(np.sqrt(lat_range**2 + lon_range**2), 4)
        
        # Speed statistics
        if 'SOG' in df.columns:
            sog_valid = vessel_data['SOG'].dropna()
            if len(sog_valid) > 0:
                stats['sog_mean'] = round(sog_valid.mean(), 2)
                stats['sog_max'] = round(sog_valid.max(), 2)
                stats['sog_min'] = round(sog_valid.min(), 2)
                stats['sog_std'] = round(sog_valid.std(), 2)
            else:
                stats['sog_mean'] = np.nan
                stats['sog_max'] = np.nan
                stats['sog_min'] = np.nan
                stats['sog_std'] = np.nan
        
        # Most common navigational status
        if 'Navigational status' in df.columns:
            nav_status = vessel_data['Navigational status'].mode()
            stats['most_common_nav_status'] = nav_status[0] if len(nav_status) > 0 else np.nan
        
        temporal_stats.append(stats)
    
    # Merge temporal stats into static_df
    temporal_df = pd.DataFrame(temporal_stats)
    static_df = static_df.merge(temporal_df, on='MMSI', how='left')
    
    # Reorder columns: MMSI first, then temporal stats, then static info
    temporal_cols = [
        'message_count', 'first_seen', 'last_seen', 'time_in_area_hours', 
        'avg_messages_per_hour', 'unique_days', 'activity_pattern',
        'lat_center', 'lon_center', 'geographic_spread_deg',
        'sog_mean', 'sog_max', 'sog_min', 'sog_std',
        'most_common_nav_status'
    ]
    
    # Filter to only existing columns
    temporal_cols = [c for c in temporal_cols if c in static_df.columns]
    static_cols = [c for c in static_df.columns if c not in temporal_cols and c != 'MMSI']
    
    column_order = ['MMSI'] + temporal_cols + static_cols
    static_df = static_df[column_order]
    
    # ==========================================
    # 3. CREATE DYNAMIC DATAFRAME
    # ==========================================
    available_dynamic = [col for col in DYNAMIC_COLUMNS if col in df.columns]
    dynamic_df = df[available_dynamic].copy()
    
    # ==========================================
    # 4. REPORT
    # ==========================================
    if verbose_mode:
        print(f"\n✅ Split complete:")
        print(f"   Static:  {len(static_df):,} unique vessels with {len(static_df.columns)} columns")
        print(f"   Dynamic: {len(dynamic_df):,} AIS messages with {len(dynamic_df.columns)} columns")
        
        # Activity pattern distribution
        if 'activity_pattern' in static_df.columns:
            print(f"\n📊 Activity Patterns:")
            pattern_counts = static_df['activity_pattern'].value_counts()
            for pattern, count in pattern_counts.items():
                pct = count / len(static_df) * 100
                print(f"   {pattern:<15}: {count:>5,} vessels ({pct:>5.1f}%)")
        
        # Time in area statistics
        if 'time_in_area_hours' in static_df.columns:
            print(f"\n⏰ Time in Area Statistics:")
            print(f"   Mean:   {static_df['time_in_area_hours'].mean():>8.2f} hours")
            print(f"   Median: {static_df['time_in_area_hours'].median():>8.2f} hours")
            print(f"   Min:    {static_df['time_in_area_hours'].min():>8.2f} hours")
            print(f"   Max:    {static_df['time_in_area_hours'].max():>8.2f} hours")
        
    # Check for conflicts in static data
    conflict_cols = []
    for col in agg_cols:
        if static_df[col].astype(str).str.contains(sep, regex=False).any():
            n_conflicts = static_df[col].astype(str).str.contains(sep, regex=False).sum()
            conflict_cols.append(f"{col} ({n_conflicts})")
    
    if conflict_cols and verbose_mode:
        print(f"\n⚠️  Static conflicts: {', '.join(conflict_cols)}")
    
    return static_df, dynamic_df

