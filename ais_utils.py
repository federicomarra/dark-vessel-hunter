import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from typing import Optional
import math
import config

cable_points = config.CABLE_POINTS

def merge_on_mmsi(static_df, dynamic_df):
    """
    Merges two DataFrames on the 'MMSI' column.

    Parameters:
    - static_df: DataFrame containing static vessel information.
    - dynamic_df: DataFrame containing dynamic AIS messages.

    Returns:
    - Merged DataFrame on 'MMSI'.
    """
    if 'MMSI' not in static_df.columns or 'MMSI' not in dynamic_df.columns:
        raise ValueError("Both DataFrames must contain an 'MMSI' column.")
    return pd.merge(static_df, dynamic_df, on='MMSI', how='inner')

def get_random_ships(df_static: pd.DataFrame, n: int = 10, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Return n random rows (ships) from df_static.

    Parameters:
      - df_static: DataFrame to sample from.
      - n: number of random ships to return (default 10).
      - seed: optional random seed for reproducibility.

    Returns:
      - DataFrame with n random rows (reset index).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n > len(df_static):
        raise ValueError(f"Requested n={n} but only {len(df_static)} rows available.")
    return df_static.sample(n=n, random_state=seed).reset_index(drop=True)

def get_one_boat_per_ship_type(df_static):
    """
    Selects one random boat for every different ship type from the given DataFrame.

    Parameters:
    - df_static: DataFrame with a 'Ship type' column.

    Returns:
    - DataFrame containing one random row per unique ship type.
    """
    if 'Ship type' not in df_static.columns:
        raise ValueError("The DataFrame must contain a 'Ship type' column.")
    return df_static.groupby('Ship type').apply(lambda x: x.sample(n=1)).reset_index(drop=True)

def ais_df_summary_split(static_df, dynamic_df, show_plots=False):
    """
    Comprehensive summary statistics for AIS dataset using split static/dynamic DataFrames.
    
    Args:
        static_df: Static vessel information DataFrame (from split_static_dynamic)
        dynamic_df: Dynamic trajectory DataFrame (from split_static_dynamic)
        show_plots: If True, display distribution plots (requires matplotlib)
    
    Returns:
        Dictionary with summary statistics
    """
    
    print("="*80)
    print("📊 AIS DATASET SUMMARY (Static + Dynamic)")
    print("="*80)
    
    summary = {}
    
    # ==========================================
    # 1. BASIC STATISTICS
    # ==========================================
    print("\n🔢 BASIC STATISTICS:")
    print("-" * 80)
    
    total_messages = len(dynamic_df)
    unique_vessels = len(static_df)
    
    summary['total_messages'] = total_messages
    summary['unique_vessels'] = unique_vessels
    summary['avg_messages_per_vessel'] = total_messages / unique_vessels if unique_vessels > 0 else 0
    
    print(f"  Total AIS messages:        {total_messages:>15,}")
    print(f"  Unique vessels (MMSI):     {unique_vessels:>15,}")
    print(f"  Avg messages per vessel:   {total_messages/unique_vessels:>15,.1f}")
    
    # Time range (from dynamic data)
    if 'Timestamp' in dynamic_df.columns:
        dynamic_df['Timestamp'] = pd.to_datetime(dynamic_df['Timestamp'], errors='coerce')
        time_min = dynamic_df['Timestamp'].min()
        time_max = dynamic_df['Timestamp'].max()
        time_span = (time_max - time_min).total_seconds() / 3600  # hours
        
        summary['time_range_start'] = time_min
        summary['time_range_end'] = time_max
        summary['time_span_hours'] = time_span
        
        print(f"  Time range start:          {time_min}")
        print(f"  Time range end:            {time_max}")
        print(f"  Time span:                 {time_span:>15,.1f} hours ({time_span/24:.1f} days)")
    
    # Memory usage
    memory_static = static_df.memory_usage(deep=True).sum() / 1024**2
    memory_dynamic = dynamic_df.memory_usage(deep=True).sum() / 1024**2
    memory_total = memory_static + memory_dynamic
    
    summary['memory_usage_mb'] = {
        'static': memory_static,
        'dynamic': memory_dynamic,
        'total': memory_total
    }
    
    print(f"  Memory usage (static):     {memory_static:>15,.1f} MB")
    print(f"  Memory usage (dynamic):    {memory_dynamic:>15,.1f} MB")
    print(f"  Memory usage (total):      {memory_total:>15,.1f} MB")
    
    # ==========================================
    # 2. GEOGRAPHIC COVERAGE
    # ==========================================
    if 'Latitude' in dynamic_df.columns and 'Longitude' in dynamic_df.columns:
        print("\n🌍 GEOGRAPHIC COVERAGE:")
        print("-" * 80)
        
        lat_min = dynamic_df['Latitude'].min()
        lat_max = dynamic_df['Latitude'].max()
        lon_min = dynamic_df['Longitude'].min()
        lon_max = dynamic_df['Longitude'].max()
        
        summary['bbox'] = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        print(f"  Latitude range:            {lat_min:.4f}°N to {lat_max:.4f}°N")
        print(f"  Longitude range:           {lon_min:.4f}°E to {lon_max:.4f}°E")
        print(f"  Bounding box area:         ~{(lat_max-lat_min)*(lon_max-lon_min):.2f} deg²")
    
    # ==========================================
    # 3. VESSEL PRESENCE & ACTIVITY PATTERNS
    # ==========================================
    if 'activity_pattern' in static_df.columns:
        print("\n🕒 VESSEL PRESENCE & ACTIVITY PATTERNS:")
        print("-" * 80)
        
        # Activity pattern distribution
        pattern_counts = static_df['activity_pattern'].value_counts(dropna=False)
        pattern_pcts = static_df['activity_pattern'].value_counts(normalize=True, dropna=False) * 100
        
        summary['activity_patterns'] = pattern_counts.to_dict()
        
        print(f"  {'Pattern':<20} {'Vessels':<12} {'Percentage':<12} {'Bar'}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*30}")
        
        for pattern, count in pattern_counts.items():
            pct = pattern_pcts[pattern]
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            pattern_name = "Unknown" if pd.isna(pattern) else str(pattern)
            print(f"  {pattern_name:<20} {count:>10,}   {pct:>5.1f}%       {bar}")
        
        print(f"  {'-'*20} {pattern_counts.sum():>10,}   100.0%")
    
    # Time in area statistics
    if 'time_in_area_hours' in static_df.columns:
        time_stats = static_df['time_in_area_hours'].describe()
        
        summary['time_in_area_stats'] = {
            'mean': float(time_stats['mean']),
            'median': float(time_stats['50%']),
            'std': float(time_stats['std']),
            'min': float(time_stats['min']),
            'max': float(time_stats['max']),
            'q25': float(time_stats['25%']),
            'q75': float(time_stats['75%'])
        }
        
        print(f"\n  Time in area statistics:")
        print(f"    Mean:       {time_stats['mean']:>12,.2f} hours")
        print(f"    Median:     {time_stats['50%']:>12,.2f} hours")
        print(f"    Std Dev:    {time_stats['std']:>12,.2f} hours")
        print(f"    Min:        {time_stats['min']:>12,.2f} hours")
        print(f"    Q25:        {time_stats['25%']:>12,.2f} hours")
        print(f"    Q75:        {time_stats['75%']:>12,.2f} hours")
        print(f"    Max:        {time_stats['max']:>12,.2f} hours")
        
        # Duration buckets with percentages
        print(f"\n  Duration distribution:")
        buckets = [
            ('<1 hour (transit)', 0, 1),
            ('1-6 hours (short)', 1, 6),
            ('6-24 hours (medium)', 6, 24),
            ('1-7 days', 24, 168),
            ('>7 days (long)', 168, float('inf'))
        ]
        
        for label, min_h, max_h in buckets:
            count = ((static_df['time_in_area_hours'] >= min_h) & 
                    (static_df['time_in_area_hours'] < max_h)).sum()
            pct = count / len(static_df) * 100
            bar_length = int(pct / 2)
            bar = "▓" * bar_length
            print(f"    {label:<25} {count:>6,} ({pct:>5.1f}%)  {bar}")
    
    # Unique days distribution
    if 'unique_days' in static_df.columns:
        print(f"\n  Unique days in area:")
        days_dist = static_df['unique_days'].value_counts().sort_index()
        for days, count in days_dist.head(10).items():
            pct = count / len(static_df) * 100
            print(f"    {days:2d} day(s):  {count:>6,} vessels ({pct:>5.1f}%)")
        if len(days_dist) > 10:
            other = days_dist.iloc[10:].sum()
            other_pct = other / len(static_df) * 100
            print(f"    >10 days:  {other:>6,} vessels ({other_pct:>5.1f}%)")
    
    # ==========================================
    # 4. CATEGORICAL DISTRIBUTIONS (from static)
    # ==========================================
    categorical_columns = [
        'Ship type',
        'Cargo type',
        'Type of position fixing device',
    ]
    
    summary['categorical_distributions'] = {}
    
    for col in categorical_columns:
        if col in static_df.columns:
            print(f"\n📋 {col.upper()} DISTRIBUTION:")
            print("-" * 80)
            
            counts = static_df[col].value_counts(dropna=False)
            percentages = static_df[col].value_counts(normalize=True, dropna=False) * 100
            
            summary['categorical_distributions'][col] = counts.to_dict()
            
            # Display top 10
            for i, (category, count) in enumerate(counts.head(10).items()):
                pct = percentages.iloc[i]
                bar_length = int(pct / 2)
                bar = "█" * bar_length
                cat_name = "Missing/Unknown" if pd.isna(category) else str(category)
                print(f"  {cat_name:<30} {count:>10,} ({pct:>5.1f}%)  {bar}")
            
            if len(counts) > 10:
                other_count = counts.iloc[10:].sum()
                other_pct = percentages.iloc[10:].sum()
                print(f"  {'Other categories':<30} {other_count:>10,} ({other_pct:>5.1f}%)")
            
            print(f"  {'─'*30} {counts.sum():>10,} (100.0%)")
            print(f"  Unique values: {static_df[col].nunique()}")
    
    # Navigational status (from dynamic data)
    if 'Navigational status' in dynamic_df.columns:
        print(f"\n📋 NAVIGATIONAL STATUS DISTRIBUTION (messages):")
        print("-" * 80)
        
        counts = dynamic_df['Navigational status'].value_counts(dropna=False)
        percentages = dynamic_df['Navigational status'].value_counts(normalize=True, dropna=False) * 100
        
        summary['categorical_distributions']['Navigational status'] = counts.to_dict()
        
        for i, (category, count) in enumerate(counts.head(10).items()):
            pct = percentages.iloc[i]
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            cat_name = "Missing/Unknown" if pd.isna(category) else str(category)
            print(f"  {cat_name:<30} {count:>10,} ({pct:>5.1f}%)  {bar}")
        
        if len(counts) > 10:
            other_count = counts.iloc[10:].sum()
            other_pct = percentages.iloc[10:].sum()
            print(f"  {'Other categories':<30} {other_count:>10,} ({other_pct:>5.1f}%)")
    
    # ==========================================
    # 5. NUMERIC STATISTICS (from dynamic)
    # ==========================================
    numeric_columns = ['SOG', 'COG', 'Heading', 'ROT', 'Draught']
    available_numeric = [col for col in numeric_columns if col in dynamic_df.columns]
    
    if available_numeric:
        print("\n📈 NUMERIC STATISTICS (from messages):")
        print("-" * 80)
        
        summary['numeric_statistics'] = {}
        
        for col in available_numeric:
            stats = dynamic_df[col].describe()
            summary['numeric_statistics'][col] = stats.to_dict()
            
            print(f"\n  {col}:")
            print(f"    Count:      {stats['count']:>12,.0f}")
            print(f"    Mean:       {stats['mean']:>12,.2f}")
            print(f"    Std:        {stats['std']:>12,.2f}")
            print(f"    Min:        {stats['min']:>12,.2f}")
            print(f"    25%:        {stats['25%']:>12,.2f}")
            print(f"    Median:     {stats['50%']:>12,.2f}")
            print(f"    75%:        {stats['75%']:>12,.2f}")
            print(f"    Max:        {stats['max']:>12,.2f}")
            print(f"    Missing:    {dynamic_df[col].isna().sum():>12,} ({dynamic_df[col].isna().sum()/len(dynamic_df)*100:.1f}%)")
    
    # Speed statistics from static (aggregated)
    if 'sog_mean' in static_df.columns:
        print(f"\n  Average SOG per vessel:")
        sog_stats = static_df['sog_mean'].describe()
        print(f"    Mean of means:  {sog_stats['mean']:>12,.2f} m/s")
        print(f"    Median:         {sog_stats['50%']:>12,.2f} m/s")
    
    # ==========================================
    # 6. DATA QUALITY
    # ==========================================
    print("\n🔍 DATA QUALITY:")
    print("-" * 80)
    
    # Static data quality
    print("  Static DataFrame:")
    missing_static = static_df.isnull().sum()
    missing_static_pct = (missing_static / len(static_df) * 100).round(2)
    
    for col in static_df.columns:
        if missing_static[col] > 0:
            print(f"    {col:<28} {missing_static[col]:>8,}   {missing_static_pct[col]:>6.1f}%")
    
    # Dynamic data quality
    print("\n  Dynamic DataFrame:")
    missing_dynamic = dynamic_df.isnull().sum()
    missing_dynamic_pct = (missing_dynamic / len(dynamic_df) * 100).round(2)
    
    for col in dynamic_df.columns[:10]:  # Show first 10 columns
        if missing_dynamic[col] > 0:
            print(f"    {col:<28} {missing_dynamic[col]:>8,}   {missing_dynamic_pct[col]:>6.1f}%")
    
    # Completeness scores
    completeness_static = (1 - static_df.isnull().sum().sum() / (len(static_df) * len(static_df.columns))) * 100
    completeness_dynamic = (1 - dynamic_df.isnull().sum().sum() / (len(dynamic_df) * len(dynamic_df.columns))) * 100
    
    summary['data_quality'] = {
        'static_completeness': completeness_static,
        'dynamic_completeness': completeness_dynamic
    }
    
    print(f"\n  Static completeness:       {completeness_static:>6.1f}%")
    print(f"  Dynamic completeness:      {completeness_dynamic:>6.1f}%")
    
    # ==========================================
    # 7. VESSEL MESSAGE STATISTICS
    # ==========================================
    print("\n🚢 VESSEL MESSAGE STATISTICS:")
    print("-" * 80)
    
    if 'message_count' in static_df.columns:
        msg_stats = static_df['message_count'].describe()
        
        summary['message_statistics'] = {
            'min': int(msg_stats['min']),
            'max': int(msg_stats['max']),
            'mean': float(msg_stats['mean']),
            'median': float(msg_stats['50%']),
            'q25': float(msg_stats['25%']),
            'q75': float(msg_stats['75%'])
        }
        
        print(f"  Messages per vessel:")
        print(f"    Min:        {msg_stats['min']:>12,.0f}")
        print(f"    Q25:        {msg_stats['25%']:>12,.0f}")
        print(f"    Median:     {msg_stats['50%']:>12,.0f}")
        print(f"    Mean:       {msg_stats['mean']:>12,.1f}")
        print(f"    Q75:        {msg_stats['75%']:>12,.0f}")
        print(f"    Max:        {msg_stats['max']:>12,.0f}")
        
        # Top 10 most active vessels
        print("\n  Top 10 most active vessels:")
        top_vessels = static_df.nlargest(10, 'message_count')[['MMSI', 'Name', 'Ship type', 'message_count', 'time_in_area_hours']]
        for i, (_, row) in enumerate(top_vessels.iterrows(), 1):
            name = row['Name'] if pd.notna(row['Name']) else 'Unknown'
            ship_type = row['Ship type'] if pd.notna(row['Ship type']) else 'Unknown'
            print(f"    {i:2d}. MMSI {row['MMSI']}: {row['message_count']:>8,} msgs, {row['time_in_area_hours']:>6.1f}h ({name[:20]}, {ship_type})")
    
    # ==========================================
    # 8. TEMPORAL PATTERNS
    # ==========================================
    if 'Timestamp' in dynamic_df.columns:
        print("\n⏰ TEMPORAL PATTERNS:")
        print("-" * 80)
        
        dynamic_df['hour'] = dynamic_df['Timestamp'].dt.hour
        dynamic_df['day_of_week'] = dynamic_df['Timestamp'].dt.day_name()
        dynamic_df['date'] = dynamic_df['Timestamp'].dt.date
        
        # Hourly distribution
        print("\n  Messages by hour of day:")
        hourly = dynamic_df.groupby('hour').size()
        for hour in range(24):
            count = hourly.get(hour, 0)
            pct = count / len(dynamic_df) * 100
            bar_length = int(pct / 0.5)
            bar = "▓" * bar_length
            print(f"    {hour:02d}:00  {count:>8,} ({pct:>4.1f}%)  {bar}")
        
        # Daily distribution
        print("\n  Messages by day of week:")
        daily = dynamic_df['day_of_week'].value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in daily.index:
                count = daily[day]
                pct = count / len(dynamic_df) * 100
                bar_length = int(pct / 0.5)
                bar = "▓" * bar_length
                print(f"    {day:<10} {count:>8,} ({pct:>4.1f}%)  {bar}")
    
    # ==========================================
    # 9. ENHANCED PLOTS
    # ==========================================
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            
            # Create 3x2 subplot grid
            fig, axes = plt.subplots(3, 2, figsize=(18, 14))
            fig.suptitle('AIS Dataset Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
            
            # Plot 1: Time in area distribution
            if 'time_in_area_hours' in static_df.columns:
                time_data = static_df['time_in_area_hours'].dropna()
                axes[0, 0].hist(time_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                axes[0, 0].set_title('Time in Area Distribution', fontweight='bold')
                axes[0, 0].set_xlabel('Hours')
                axes[0, 0].set_ylabel('Number of Vessels')
                axes[0, 0].set_yscale('log')
                axes[0, 0].grid(alpha=0.3)
            
            # Plot 2: Activity Pattern
            if 'activity_pattern' in static_df.columns:
                pattern_counts = static_df['activity_pattern'].value_counts()
                colors = plt.cm.Set3(range(len(pattern_counts)))
                axes[0, 1].bar(range(len(pattern_counts)), pattern_counts.values, color=colors, edgecolor='black')
                axes[0, 1].set_xticks(range(len(pattern_counts)))
                axes[0, 1].set_xticklabels(pattern_counts.index, rotation=45, ha='right')
                axes[0, 1].set_title('Activity Pattern Distribution', fontweight='bold')
                axes[0, 1].set_ylabel('Number of Vessels')
                axes[0, 1].grid(alpha=0.3, axis='y')
            
            # Plot 3: SOG Distribution
            if 'SOG' in dynamic_df.columns:
                sog_data = dynamic_df['SOG'].dropna()
                axes[1, 0].hist(sog_data, bins=50, edgecolor='black', alpha=0.7, color='seagreen')
                axes[1, 0].axvline(sog_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {sog_data.median():.1f}')
                axes[1, 0].set_title('Speed Over Ground (SOG) Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('SOG (m/s)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
            
            # Plot 4: Messages per Vessel
            if 'message_count' in static_df.columns:
                axes[1, 1].hist(static_df['message_count'], bins=50, edgecolor='black', alpha=0.7, color='mediumpurple')
                axes[1, 1].set_title('Messages per Vessel Distribution', fontweight='bold')
                axes[1, 1].set_xlabel('Number of messages')
                axes[1, 1].set_ylabel('Number of vessels')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(alpha=0.3, which='both')
            
            # Plot 5: Ship Type
            if 'Ship type' in static_df.columns:
                ship_type = static_df['Ship type'].value_counts().head(10)
                colors = plt.cm.Paired(range(len(ship_type)))
                bars = axes[2, 0].barh(range(len(ship_type)), ship_type.values, color=colors, edgecolor='black')
                axes[2, 0].set_yticks(range(len(ship_type)))
                axes[2, 0].set_yticklabels([str(x)[:25] for x in ship_type.index], fontsize=9)
                axes[2, 0].set_xlabel('Count')
                axes[2, 0].set_title('Top 10 Ship Types', fontweight='bold')
                axes[2, 0].invert_yaxis()
                axes[2, 0].grid(alpha=0.3, axis='x')
            
            # Plot 6: Navigational Status
            if 'Navigational status' in dynamic_df.columns:
                nav_status = dynamic_df['Navigational status'].value_counts().head(10)
                colors = plt.cm.Set3(range(len(nav_status)))
                bars = axes[2, 1].barh(range(len(nav_status)), nav_status.values, color=colors, edgecolor='black')
                axes[2, 1].set_yticks(range(len(nav_status)))
                axes[2, 1].set_yticklabels([str(x)[:25] for x in nav_status.index], fontsize=9)
                axes[2, 1].set_xlabel('Count')
                axes[2, 1].set_title('Top 10 Navigational Status', fontweight='bold')
                axes[2, 1].invert_yaxis()
                axes[2, 1].grid(alpha=0.3, axis='x')
            
            plt.tight_layout()
            # plt.savefig('ais_summary_plots_split.png', dpi=150, bbox_inches='tight')
            # print("\n📊 Plots saved to: ais_summary_plots_split.png")
            plt.show()
            
        except ImportError as e:
            print(f"\n⚠️  Error creating plots: {e}")
            print("   Install matplotlib and seaborn: pip install matplotlib seaborn")
    
    print("\n" + "="*80)
    print("✅ Summary complete!")
    print("="*80)
    
    return summary

def build_polyline(points_latlon):
    """
    points_latlon: lista di (lat, lon) IN GRADI, nell'ordine in cui
                   vuoi che vengano collegati.
    Ritorna: shapely.geometry.LineString
    """
    if len(points_latlon) < 2:
        raise ValueError("Servono almeno 2 punti per costruire una spezzata.")
    
    # Shapely vuole (x, y) = (lon, lat)
    coords = [(lon, lat) for (lat, lon) in points_latlon]
    return LineString(coords)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_closest_point_on_segment(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon):
    """
    Finds the closest point on segment AB to point P.
    Uses vector projection on lat/lon (sufficiently accurate for local scale).
    """
    # Vector form relative to A
    x = p_lon - a_lon
    y = p_lat - a_lat
    dx = b_lon - a_lon
    dy = b_lat - a_lat
    
    if dx == 0 and dy == 0:
        return a_lat, a_lon

    # Calculate the projection scalar 't'
    # t represents how far along the line segment the closest point is (0 to 1)
    t = (x * dx + y * dy) / (dx*dx + dy*dy)
    
    # Clamp t to the segment range [0, 1]
    # If t < 0, closest is point A; if t > 1, closest is point B
    t = max(0, min(1, t))
    
    # The coordinates of the closest point on the line
    closest_lat = a_lat + t * dy
    closest_lon = a_lon + t * dx
    
    return closest_lat, closest_lon

def get_min_distance_to_cables(vessel_lat, vessel_lon, cables_dict):
    """
    Returns:
    1. Minimum distance in meters.
    2. Name of the nearest cable.
    3. Coordinates (lat, lon) of the specific point on that cable.
    """
    min_dist = float('inf')
    nearest_cable = None
    nearest_point = (None, None)

    for cable_name, points in cables_dict.items():
        # Iterate through every segment (from point i to point i+1)
        for i in range(len(points) - 1):
            p1_lat, p1_lon = points[i]
            p2_lat, p2_lon = points[i+1]
            
            # Find the closest point on THIS segment
            c_lat, c_lon = get_closest_point_on_segment(vessel_lat, vessel_lon, p1_lat, p1_lon, p2_lat, p2_lon)
            
            # Calculate real physical distance to that projected point
            dist = haversine_distance(vessel_lat, vessel_lon, c_lat, c_lon)
            
            if dist < min_dist:
                min_dist = dist
                nearest_cable = cable_name
                nearest_point = (c_lat, c_lon)

    return min_dist, nearest_cable, nearest_point

def analyze_cable_risks(df, cables_dict, lat_col='Lat', lon_col='Lon'):
    """
    Calculates the distance of every vessel position to the nearest cable.
    
    Args:
        df: Input DataFrame
        cables_dict: The dictionary of cable points
        lat_col: Name of the column containing Latitude
        lon_col: Name of the column containing Longitude
        
    Returns:
        A new DataFrame with MMSI, Date, Distance, and the exact point on the cable.
    """
    
    # 1. Pre-allocate lists for speed
    distances = []
    cable_names = []
    cable_lats = []
    cable_lons = []
    mmsis = []
    dates = []
    
    # 2. Extract numpy arrays for fast iteration
    #    (Using .values is much faster than .iterrows)
    lats = df[lat_col].values
    lons = df[lon_col].values
    mmsi_list = df['MMSI'].values
    date_list = df['Date'].values
    
    print(f"Processing {len(df)} rows for cable proximity...")
    
    # 3. Iterate with a progress bar
    for lat, lon, mmsi, date in tqdm(zip(lats, lons, mmsi_list, date_list), total=len(df)):
        
        # Call your existing function
        dist, name, (c_lat, c_lon) = get_min_distance_to_cables(lat, lon, cables_dict)
        
        # Store results
        distances.append(dist)
        cable_names.append(name)
        cable_lats.append(c_lat)
        cable_lons.append(c_lon)
        mmsis.append(mmsi)
        dates.append(date)

    # 4. Build the Result DataFrame
    risk_df = pd.DataFrame({
        'MMSI': mmsis,
        'Date': dates,
        'distance_to_cable_m': distances,
        'nearest_cable': cable_names,
        'cable_point_lat': cable_lats,  # The specific "endangered" point on the cable
        'cable_point_lon': cable_lons
    })
    
    return risk_df
