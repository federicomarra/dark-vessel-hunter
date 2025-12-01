import folium
from folium import LayerControl
import pandas as pd
import numpy as np
from branca.element import Template, MacroElement
from folium.plugins import HeatMap
from typing import Sequence, Optional



def create_ship_path_html(df, mmsi, out_html=None, center=None, n_points=24, zoom_start=11): 
    """
    Creates a Folium map for a specific vessel (MMSI) from a single merged DataFrame.
    
    Parameters:
    - df: The merged DataFrame containing both dynamic (Lat/Lon/Time) and static (Name/Type) columns.
    - mmsi: The MMSI of the vessel to plot.
    - out_html: (Optional) Filename to save the HTML map.
    - center: (Optional) [lat, lon] to center the map.
    - n_points: Number of markers to plot along the path (to avoid clutter).
    - zoom_start: Initial zoom level.
    """
    
    # 1. Filter data for the specific MMSI
    #    Convert to string to ensure matching works even if input is int/str
    mmsi_str = str(mmsi)
    vessel = df[df['MMSI'].astype(str) == mmsi_str].copy()
    
    if vessel.empty:
        print(f"No data found for MMSI {mmsi}")
        return None

    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(vessel['Timestamp']):
        vessel['Timestamp'] = pd.to_datetime(vessel['Timestamp'])
        
    # Sort by time
    vessel = vessel.sort_values('Timestamp').reset_index(drop=True)

    # 2. Extract Static Info (from the first row)
    #    Use .get() to avoid crashes if columns are missing
    first_row = vessel.iloc[0]
    
    ship_name = str(first_row.get('Name', 'Unknown'))
    callsign = str(first_row.get('Callsign', '-'))
    imo = str(first_row.get('IMO', '-'))
    ship_type = str(first_row.get('Ship type', first_row.get('ShipType', '-')))
    length = str(first_row.get('Length', '-'))
    width = str(first_row.get('Width', '-'))

    # 3. Sampling Logic (Select only n_points for markers)
    times = vessel['Timestamp']
    
    if len(vessel) > 1:
        total_hours = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0
    else:
        total_hours = 0
    
    selected_indices = [0] # Always include the start
    
    if total_hours > 0 and n_points > 1:
        step = total_hours / float(n_points)
        prev_time = times.iloc[0]
        
        for _ in range(n_points - 1):
            target_time = prev_time + pd.Timedelta(hours=step)
            # Find the first index where time >= target_time
            matches = np.where(times.values >= target_time)[0]
            
            if matches.size == 0:
                break
                
            idx = int(matches[0])
            
            # Prevent adding the same index twice
            if idx > selected_indices[-1]:
                selected_indices.append(idx)
                prev_time = times.iloc[idx]
    else:
        # If very short duration or few points, take them all
        if len(vessel) > 1:
            selected_indices = list(range(len(vessel)))

    # Always include the last point
    if selected_indices[-1] != len(vessel) - 1:
        selected_indices.append(len(vessel) - 1)

    # Safety cap if something went wrong
    if len(selected_indices) > n_points + 5:
        selected_indices = np.linspace(0, len(vessel)-1, n_points).astype(int).tolist()

    # Create the subset for markers
    subset = vessel.iloc[selected_indices].reset_index(drop=True)
    
    # 4. Create Map
    if center is None:
        center_use = [vessel['Latitude'].mean(), vessel['Longitude'].mean()]
    else:
        center_use = center

    m = folium.Map(location=center_use, zoom_start=zoom_start, tiles="CartoDB positron")

    # 5. Draw the Path (PolyLine) using ALL points (for high detail)
    path_coords = vessel[['Latitude', 'Longitude']].values.tolist()
    folium.PolyLine(path_coords, color='blue', weight=3, opacity=0.7).add_to(m)

    # 6. Add Markers (using only the sampled subset)
    for i, row in subset.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        ts = row['Timestamp']
        sog = row.get('SOG', '-')
        cog = row.get('COG', '-')
        
        # HTML for the Popup
        info_html = f"""
        <b>Time:</b> {ts}<br>
        <b>SOG:</b> {sog} kn<br>
        <b>COG:</b> {cog}°
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.9,
            popup=folium.Popup(info_html, max_width=250),
            tooltip=folium.Tooltip(f"Time: {ts} | Speed: {sog} kn", sticky=True)
        ).add_to(m)

    # 7. Add Static Info Legend (Top-Right)
    legend_html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed; 
        top: 10px; right: 10px; width: 200px; height: auto;
        z-index:9999; background-color: white; opacity: 0.9;
        padding: 10px; border: 2px solid grey; border-radius: 5px;
        font-family: sans-serif; font-size: 12px;
    ">
        <b>Vessel Info (MMSI: {mmsi})</b><br><hr>
        <b>Name:</b> {ship_name}<br>
        <b>Callsign:</b> {callsign}<br>
        <b>IMO:</b> {imo}<br>
        <b>Type:</b> {ship_type}<br>
        <b>Size:</b> {length} x {width} m
    </div>
    {{% endmacro %}}
    """
    
    macro = MacroElement()
    macro._template = Template(legend_html)
    m.get_root().add_child(macro)

    # 8. Save output
    if out_html:
        m.save(out_html)
        print(f"Map saved to: {out_html}")
        
    return m



def create_all_ships_paths_html(df, out_html="all_ships_map.html",
                                center=None, zoom_start=9,
                                n_points_per_ship=24):
    """
    Creates a Folium map with the paths of ALL ships (all MMSIs) 
    present in the dataframe.

    Parameters:
    - df: DataFrame with at least columns: 'MMSI', 'Latitude', 'Longitude', 'Timestamp'
      (optional: 'Name', 'Ship type'/'ShipType')
    - out_html: name of the output HTML file.
    - center: [lat, lon] to center the map. If None, uses the mean of the dataset.
    - zoom_start: initial zoom level.
    - n_points_per_ship: number of "sampled" points for markers for each ship.
    """

    if df.empty:
        print("Empty DataFrame: no ships to plot.")
        return None

    # Ensure Timestamp is datetime
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Center of the map
    if center is None:
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()
        center_use = [center_lat, center_lon]
    else:
        center_use = center

    m = folium.Map(location=center_use, zoom_start=zoom_start, tiles="CartoDB positron")

    # Base color palette to cycle through for ships
    colors = [
        "blue", "red", "green", "purple", "orange",
        "darkred", "lightred", "beige", "darkblue", "darkgreen",
        "cadetblue", "darkpurple", "white", "pink", "lightblue",
        "lightgreen", "gray", "black", "lightgray"
    ]

    # Group by MMSI
    df['MMSI_str'] = df['MMSI'].astype(str)
    unique_mmsi = df['MMSI_str'].unique()

    print(f"Found {len(unique_mmsi)} MMSIs in the DataFrame.")
    for idx, mmsi_str in enumerate(unique_mmsi):
        vessel = df[df['MMSI_str'] == mmsi_str].copy()
        if vessel.empty:
            continue

        # Sort by time if available
        if 'Timestamp' in vessel.columns:
            vessel = vessel.sort_values('Timestamp')

        # Choose a color for this ship
        color = colors[idx % len(colors)]

        # Retrieve some static info from the first row (if available)
        first_row = vessel.iloc[0]
        ship_name = str(first_row.get('Name', 'Unknown'))
        ship_type = str(first_row.get('Ship type', first_row.get('ShipType', '-')))

        # Create a FeatureGroup for each ship, so they can be toggled on/off
        fg = folium.FeatureGroup(name=f"{ship_name} ({mmsi_str})")

        # 1) Polyline with ALL points
        path_coords = vessel[['Latitude', 'Longitude']].values.tolist()
        if len(path_coords) < 2:
            # little movement, skip the polyline but still show the marker
            pass
        else:
            folium.PolyLine(
                path_coords,
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"MMSI: {mmsi_str} | {ship_name}"
            ).add_to(fg)

        # 2) Markers only on some points (sampled)
        if n_points_per_ship is not None and len(vessel) > 0:
            # sample evenly spaced indices
            if len(vessel) <= n_points_per_ship:
                subset = vessel
            else:
                sample_indices = np.linspace(0, len(vessel) - 1, n_points_per_ship).astype(int)
                subset = vessel.iloc[sample_indices]

            for _, row in subset.iterrows():
                lat = row['Latitude']
                lon = row['Longitude']
                ts = row.get('Timestamp', '-')
                sog = row.get('SOG', '-')
                info_html = f"""
                <b>MMSI:</b> {mmsi_str}<br>
                <b>Name:</b> {ship_name}<br>
                <b>Type:</b> {ship_type}<br>
                <b>Time:</b> {ts}<br>
                <b>SOG:</b> {sog} kn
                """
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=folium.Popup(info_html, max_width=250),
                    tooltip=f"{ship_name} | MMSI: {mmsi_str}"
                ).add_to(fg)

        fg.add_to(m)

    # Layer control to toggle ships on/off
    folium.LayerControl(collapsed=False).add_to(m)

    # Save HTML
    if out_html:
        m.save(out_html)
        print(f"Map with all ships saved to: {out_html}")

    return m


def create_cable_risk_heatmap(df, dist_threshold=2000, out_html="cable_risk_map.html"):
    
    """
    Creates a heatmap of the SUBMARINE CABLE points, colored by how frequently
    ships pass near them.
    
    Parameters:
    - df: The dataframe containing 'distance_to_cable_m', 'cable_point_lat', 'cable_point_lon'.
    - dist_threshold: Only consider ships closer than this value (meters). 
                      Set to None to use ALL data points.
    """
    
    print("Generating Cable Risk Heatmap...")
    
    # 1. Filter by distance (Focus on actual risks)
    if dist_threshold is not None:
        risk_df = df[df['distance_to_cable_m'] <= dist_threshold].copy()
        print(f"Filtering: Kept {len(risk_df)} interaction points closer than {dist_threshold}m.")
    else:
        risk_df = df.copy()

    if risk_df.empty:
        print("No data found within the distance threshold.")
        return None

    # 2. Aggregation
    # We round the cable coordinates slightly to group very close points together
    # 4 decimal places is roughly 11 meters precision.
    risk_df['lat_round'] = risk_df['cable_point_lat'].round(4)
    risk_df['lon_round'] = risk_df['cable_point_lon'].round(4)

    # Count how many times each cable point was 'hit'
    heatmap_data = risk_df.groupby(['lat_round', 'lon_round']).size().reset_index(name='count')

    # 3. Prepare Data for Folium (Convert to standard Python floats to avoid JSON errors)
    max_val = float(heatmap_data['count'].max())
    heat_data = heatmap_data[['lat_round', 'lon_round', 'count']].astype(float).values.tolist()

    print(f"Identified {len(heatmap_data)} unique vulnerable points on the cable.")
    print(f"Highest traffic point has {max_val} encounters.")

    # 4. Create Map
    center_lat = heatmap_data['lat_round'].mean()
    center_lon = heatmap_data['lon_round'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    # Add the Heatmap Layer
    # Gradient: Blue (Low risk) -> Yellow -> Red (High risk)
    HeatMap(
        heat_data,
        name="Cable Risk Intensity",
        min_opacity=0.4,
        max_val=max_val,
        radius=15,  # Radius of the glow
        blur=10,    # Smoothness
        gradient={0.2: 'blue', 0.5: 'lime', 0.7: 'orange', 1.0: 'red'}
    ).add_to(m)

    # Optional: Add small circles for the absolute hottest spots
    # Filter for points with very high traffic (e.g., top 10%)
    top_risks = heatmap_data[heatmap_data['count'] > (max_val * 0.8)]
    
    for _, row in top_risks.iterrows():
        folium.CircleMarker(
            location=[row['lat_round'], row['lon_round']],
            radius=3,
            color='black',
            weight=1,
            fill=False,
            tooltip=f"High Risk Point: {int(row['count'])} encounters"
        ).add_to(m)

    folium.LayerControl().add_to(m)

    if out_html:
        m.save(out_html)
        print(f"Map saved to {out_html}")

    return m

def make_ais_tracks_map(
    df_list: Sequence[pd.DataFrame],
    bbox,
    polygon_coords: Optional[Sequence[tuple[float, float]]] = None,
    max_vessels: Optional[int] = None,
) -> folium.Map:
    """
    Create an interactive Folium map of AIS vessel tracks from one or more DataFrames.

    Each dataframe must contain at least:
        ['Latitude', 'Longitude', 'MMSI', 'Timestamp']

    Parameters
    ----------
    df_list : list of pd.DataFrame
        DataFrames containing AIS records.
    bbox : list or tuple
        Bounding box [lat_max, lon_min, lat_min, lon_max].
    polygon_coords : list of (lon, lat), optional
        Optional polygon defining the AOI. Drawn on the map if provided.
    max_vessels : int or None
        Maximum number of MMSI tracks to plot.
        If None: plot ALL vessels.

    Returns
    -------
    folium.Map
        An interactive map with vessel tracks.
    """

    # ======================================================================
    # 1. Combine all dataframes
    # ======================================================================
    df = pd.concat(df_list, ignore_index=True)

    # Normalize column names in case different formats appear
    rename_map = {}
    if "lat" in df.columns and "Latitude" not in df.columns:
        rename_map["lat"] = "Latitude"
    if "lon" in df.columns and "Longitude" not in df.columns:
        rename_map["lon"] = "Longitude"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Remove invalid/missing positions
    df = df.dropna(subset=["Latitude", "Longitude", "MMSI"])

    # Ensure MMSI is string and timestamp is datetime
    df["MMSI"] = df["MMSI"].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

    # Sort by vessel and time
    df = df.sort_values(["MMSI", "Timestamp"])

    # ======================================================================
    # 2. Initialize base map centered on bbox
    # ======================================================================
    lat_max, lon_min, lat_min, lon_max = bbox
    center_lat = (lat_max + lat_min) / 2
    center_lon = (lon_min + lon_max) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    # ======================================================================
    # 3. Draw AOI polygon if provided
    # ======================================================================
    if polygon_coords is not None:
        # polygon_coords = [(lon, lat), ...]
        poly_latlon = [(lat, lon) for lon, lat in polygon_coords]
        folium.Polygon(
            locations=poly_latlon,
            color="red",
            weight=2,
            fill=False,
            popup="AOI Polygon",
        ).add_to(m)

    # ======================================================================
    # 4. Select vessels to plot
    # ======================================================================
    unique_mmsi = df["MMSI"].unique()

    if max_vessels is not None and len(unique_mmsi) > max_vessels:
        print(f"⚠️ {len(unique_mmsi)} vessels found. Plotting only first {max_vessels}.")
        unique_mmsi = unique_mmsi[:max_vessels]
        df = df[df["MMSI"].isin(unique_mmsi)]

    # Create a group layer for vessel tracks
    tracks_group = folium.FeatureGroup(name="AIS Tracks")
    m.add_child(tracks_group)

    # ======================================================================
    # 5. Draw tracks vessel-by-vessel
    # ======================================================================
    for mmsi, df_vessel in df.groupby("MMSI"):
        coords = list(zip(df_vessel["Latitude"].values, df_vessel["Longitude"].values))
        if len(coords) < 2:
            continue

        # Polyline representing the vessel's track
        folium.PolyLine(
            locations=coords,
            weight=2,
            opacity=0.7,
            popup=f"MMSI: {mmsi} | Points: {len(coords)}",
        ).add_to(tracks_group)

        # Mark start and end
        start_lat, start_lon = coords[0]
        end_lat, end_lon = coords[-1]

        folium.CircleMarker(
            location=[start_lat, start_lon],
            radius=3,
            color="green",
            popup=f"Start MMSI: {mmsi}",
        ).add_to(tracks_group)

        folium.CircleMarker(
            location=[end_lat, end_lon],
            radius=3,
            color="red",
            popup=f"End MMSI: {mmsi}",
        ).add_to(tracks_group)

    # Add layer control
    LayerControl().add_to(m)

    return m

