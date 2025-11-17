import pandas as pd
import numpy as np
from pathlib import Path

from typing import Sequence, Optional
from collections.abc import Sequence
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import contains_xy




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




def filter_ais_df(
    df: pd.DataFrame,
    polygon_coords: Sequence[tuple[float, float]],
    allowed_mobile_types: Optional[Sequence[str]] = ("Class A", "Class B"),
    bbox: Optional[Sequence[float]] = None,
    apply_polygon_filter: bool = True,
    remove_zero_sog_vessels: bool = True,
    sog_in_knots: bool = True,
    port_locodes_path: Path = None,
    exclude_ports: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply AIS filtering steps to a DataFrame.

    Steps:
    0) Basic sanity checks and Timestamp handling (# Timestamp -> Timestamp)
    1) Optional filter by "Type of mobile" 
    2) MMSI sanity checks (length == 9 and MID in [200, 775])
    3) Drop duplicates on (Timestamp, MMSI)
    4) Optional bounding box filter
    5) Optional polygon filtering using Shapely (lon, lat)
    6) Optional removal of AIS points inside port polygons
    7) Convert SOG from knots to m/s (if sog_in_knots=True)
    8) Optional removal of ships with >90% zero SOG

    Parameters
    ----------
    df : pd.DataFrame
        Input AIS DataFrame with:
        ["Latitude", "Longitude", "MMSI", "SOG", "Timestamp" or "# Timestamp"].
    polygon_coords : Sequence[tuple[float, float]]
        Polygon vertices as (lon, lat) pairs.
    allowed_mobile_types : Sequence[str] or None
        Allowed types of mobile (e.g., Class A or B AIS transponders).
    bbox : [north_lat, west_lon, south_lat, east_lon] or None
    apply_polygon_filter : bool
    remove_zero_sog_vessels : bool
        If True, removes ships with >90% SOG==0.
    sog_in_knots : bool
        If True, convert SOG (knots → m/s).
    port_locodes_path : str or None
        Path to port_locodes.csv containing port polygons.
    exclude_ports : bool
        If True and port_locodes_path is provided, remove AIS points
        that fall inside any overlapping port polygon.
    verbose : bool
        Print filtering info.

    Returns
    -------
    pd.DataFrame
        Filtered AIS DataFrame.
    """

    df = df.copy()

    # ------------------------------------------------------------------
    # 0) Basic checks and Timestamp handling
    # ------------------------------------------------------------------
    required_columns = ["Latitude", "Longitude", "MMSI", "SOG"]

    # Timestamp can be "# Timestamp" or "Timestamp"
    if "# Timestamp" in df.columns and "Timestamp" not in df.columns:
        df = df.rename(columns={"# Timestamp": "Timestamp"})

    required_columns_ts = required_columns + ["Timestamp"]
    for col in required_columns_ts:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    df["MMSI"] = df["MMSI"].astype(str).str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if verbose:
        print(
            f" [filter_ais_df] Before filtering: {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} vessels"
        )

    # ------------------------------------------------------------------
    # 1) Type of mobile filter
    # ------------------------------------------------------------------
    if "Type of mobile" in df.columns and allowed_mobile_types is not None:
        before = len(df)
        df = df[df["Type of mobile"].isin(allowed_mobile_types)]
        if verbose:
            print(
                f" [filter_ais_df] Type filtering: {len(df):,} rows "
                f"(removed {before - len(df):,}) using {list(allowed_mobile_types)}"
            )
    elif "Type of mobile" not in df.columns and verbose:
        print(" [filter_ais_df] Warning: 'Type of mobile' column not found, skipping that filter.")

    # ------------------------------------------------------------------
    # 2) MMSI sanity filters
    # ------------------------------------------------------------------
    mmsi_str = df["MMSI"]

    mask_len = mmsi_str.str.len() == 9
    mid = mmsi_str.str[:3]
    mask_mid = mid.str.isnumeric() & mid.astype(int).between(200, 775)

    mask_valid = mask_len & mask_mid
    df = df[mask_valid].copy()

    if verbose:
        print(
            f" [filter_ais_df] MMSI filtering: {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} vessels"
        )

    # ------------------------------------------------------------------
    # 3) Remove duplicates
    # ------------------------------------------------------------------
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")

    if verbose:
        print(
            f" [filter_ais_df] Duplicate removal: {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} vessels"
        )

    # ------------------------------------------------------------------
    # 4) Optional bounding box filtering
    # ------------------------------------------------------------------
    if bbox is not None:
        north, west, south, east = bbox
        before = len(df)

        df = df[
            (df["Latitude"] <= north)
            & (df["Latitude"] >= south)
            & (df["Longitude"] >= west)
            & (df["Longitude"] <= east)
        ]

        if verbose:
            print(
                f" [filter_ais_df] BBOX filtering: {len(df):,} rows "
                f"(removed {before - len(df):,}), {df['MMSI'].nunique():,} vessels"
            )

    # ------------------------------------------------------------------
    # 5) Polygon filtering (vectorized)
    # ------------------------------------------------------------------
    main_polygon = None
    if apply_polygon_filter and polygon_coords is not None:
        main_polygon = Polygon(polygon_coords)

        lons = df["Longitude"].to_numpy()
        lats = df["Latitude"].to_numpy()
        mask_poly = contains_xy(main_polygon, lons, lats)

        before = len(df)
        df = df[mask_poly].copy()

        if verbose:
            print(
                f" [filter_ais_df] Polygon filtering: {len(df):,} rows "
                f"(removed {before - len(df):,}), {df['MMSI'].nunique():,} vessels"
            )

    # ------------------------------------------------------------------
    # 6) Remove AIS points inside port polygons (from port_locodes.csv)
    # ------------------------------------------------------------------
    if (
        exclude_ports
        and port_locodes_path is not None
        and apply_polygon_filter
        and polygon_coords is not None
    ):
        # Load the port polygons from CSV: name;locode;lon lat,lon lat,...
        ports_df = pd.read_csv(
            port_locodes_path,
            sep=";",
            header=None,
            names=["port_name", "locode", "coords"],
            engine="python"
        )

        def parse_coord_string(coord_str: str) -> list[tuple[float, float]]:
            coords: list[tuple[float, float]] = []
            for pair in str(coord_str).split(","):
                pair = pair.strip()
                if not pair:
                    continue
                parts = pair.split()
                if len(parts) != 2:
                    continue
                lon, lat = map(float, parts)
                coords.append((lon, lat))
            return coords

        ports_df["polygon"] = ports_df["coords"].apply(
            lambda s: Polygon(parse_coord_string(s))
        )

        # Intersect only with the area we're actually using
        if main_polygon is None:
            main_polygon = Polygon(polygon_coords)

        ports_df = ports_df[
            ports_df["polygon"].apply(lambda p: p.is_valid and p.intersects(main_polygon))
        ]

        if not ports_df.empty:
            ports_union = MultiPolygon(ports_df["polygon"].tolist())

            lons = df["Longitude"].to_numpy()
            lats = df["Latitude"].to_numpy()
            mask_in_ports = contains_xy(ports_union, lons, lats)

            removed_rows = int(mask_in_ports.sum())
            df = df[~mask_in_ports].copy()

            if verbose:
                print(
                    f" [filter_ais_df] Port-area removal: removed {removed_rows:,} rows "
                    f"in {len(ports_df):,} overlapping ports"
                )
        elif verbose:
            print(" [filter_ais_df] No port polygons intersect the main polygon; skipping port removal.")

    # ------------------------------------------------------------------
    # 7) SOG conversion to m/s
    # ------------------------------------------------------------------
    if sog_in_knots:
        df["SOG"] = df["SOG"].astype(float) * 0.514444

    # ------------------------------------------------------------------
    # 8) Remove vessels with >90% zero SOG
    # ------------------------------------------------------------------
    if remove_zero_sog_vessels:
        sog_zero_fraction = df.groupby("MMSI")["SOG"].apply(lambda x: (x <= 0).mean())
        bad_mmsi = sog_zero_fraction[sog_zero_fraction > 0.9].index

        df = df[~df["MMSI"].isin(bad_mmsi)]

        if verbose:
            print(
                f" [filter_ais_df] Removed >90% zero-SOG vessels: "
                f"{len(bad_mmsi):,} vessels removed"
            )

    if verbose:
        print(
            f" [filter_ais_df] Final: {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} unique vessels"
        )

    return df
