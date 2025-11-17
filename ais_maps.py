import folium
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from branca.element import Template, MacroElement
import random


def create_ship_path_html(mmsi, df_dynamic, df_static=None, out_html=None, center=None, n_points=24, zoom_start=11): 
    """
    Create a folium map HTML for a vessel's track sampled at up to `n_points`
    and with hover-tooltips showing Timestamp / SOG / COG. If `df_static` is
    provided, a small static-data legend is added to the top-right corner.

    Parameters:
    - df_dynamic: DataFrame with columns ['MMSI','Timestamp','Latitude','Longitude', ...]
                  Timestamp must be datetime dtype.
    - mmsi: str or int identifying the vessel in df_dynamic['MMSI']
    - out_html: optional output filename (default "map_{mmsi}.html")
    - center: optional [lat, lon] for initial map center; if None centroid of selected vessel points is used
    - n_points: number of points to select (default 24)
    - zoom_start: folium map zoom level
    - df_static: optional DataFrame with static vessel info (must include 'MMSI' column)

    Returns:
    - folium.Map object
    """
    # filter and sort
    vessel = df_dynamic[df_dynamic['MMSI'].astype(str) == str(mmsi)].sort_values('Timestamp').reset_index(drop=True)
    if vessel.empty:
        raise ValueError(f"No data for MMSI {mmsi}")

    times = vessel['Timestamp']
    total_hours = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0
    tot = total_hours / float(n_points) if total_hours > 0 else 0.0

    selected_pos = []
    if tot > 0:
        prev_time = times.iloc[0]
        selected_pos.append(0)
        for _ in range(n_points - 1):
            target = prev_time + pd.Timedelta(hours=tot)
            later_idx = np.where(times.values >= target)[0]
            if later_idx.size == 0:
                break
            pos = int(later_idx[0])
            if pos <= selected_pos[-1]:
                break
            selected_pos.append(pos)
            prev_time = times.iloc[pos]

    if selected_pos == [] or selected_pos[-1] != len(vessel) - 1:
        selected_pos.append(len(vessel) - 1)

    if len(selected_pos) < n_points:
        count = min(n_points, len(vessel))
        selected_pos = list(np.linspace(0, len(vessel) - 1, count).astype(int))

    selected_pos = selected_pos[:n_points]

    sel = vessel.iloc[selected_pos].reset_index(drop=True)
    coords = sel[['Latitude', 'Longitude']].values.tolist()

    # map center
    if center is None:
        center_use = [float(sel['Latitude'].mean()), float(sel['Longitude'].mean())]
    else:
        center_use = center

    # build folium map
    m = folium.Map(location=center_use, zoom_start=zoom_start)
    folium.PolyLine(coords, color='blue', weight=3, opacity=0.7).add_to(m)

    # create an invisible hover layer that shows full details on mouseover
    hover_fg = folium.FeatureGroup(name="hover_tooltips", show=True)
    for i, row in enumerate(sel.itertuples(index=False)):
        lat = getattr(row, 'Latitude')
        lon = getattr(row, 'Longitude')
        ts = getattr(row, 'Timestamp')
        sog = getattr(row, 'SOG', None)
        cog = getattr(row, 'COG', None)
        mmsi_val = getattr(row, 'MMSI', None)

        # tooltip to show on hover (use same info as popup)
        hover_tooltip = folium.Tooltip(f"MMSI: {mmsi_val}<br>Time: {ts}<br>SOG: {sog}<br>COG: {cog}", sticky=True)

        # invisible marker that only provides the hover tooltip (so clicking is not required)
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="transparent",
            fill=True,
            fill_opacity=0.0,
            opacity=0.0,
            tooltip=hover_tooltip
        ).add_to(hover_fg)
    m.add_child(hover_fg)
    for i, row in enumerate(sel.itertuples(index=False)):
        lat = getattr(row, 'Latitude')
        lon = getattr(row, 'Longitude')
        ts = getattr(row, 'Timestamp')
        sog = getattr(row, 'SOG', None)
        cog = getattr(row, 'COG', None)
        mmsi_val = getattr(row, 'MMSI', None)

        tooltip_html = f"Time: {ts}<br>SOG: {sog}<br>COG: {cog}"
        popup_html = f"MMSI: {mmsi_val}<br>Time: {ts}<br>SOG: {sog}<br>COG: {cog}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='red',
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=folium.Tooltip(tooltip_html, sticky=True)
        ).add_to(m)

        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=f"<div style='font-size:10px;color:white;background:rgba(0,0,0,0.5);padding:2px;border-radius:3px'>{i+1}</div>")
        ).add_to(m)

    # add static legend if df_static provided
    if df_static is not None:
        try:
            stat_row = df_static[df_static['MMSI'].astype(str) == str(mmsi)]
            if not stat_row.empty:
                stat = stat_row.iloc[0]
                def v(k):
                    val = stat.get(k, "")
                    return "" if pd.isna(val) else str(val)
                name = v('Name') or v('Callsign') or ""
                callsign = v('Callsign')
                imo = v('IMO')
                ship_type = v('Ship type')
                length = v('Length')
                width = v('Width')

                # create an HTML legend using branca Template
                legend_html = f"""
                {{% macro html(this, kwargs) %}}
                <div style="
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index:9999;
                    background-color: white;
                    padding:10px;
                    border:2px solid grey;
                    box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
                    font-size:12px;
                    line-height:1.4;
                ">
                  <b>Static info</b><br>
                  <b>Name:</b> {name}<br>
                  <b>Callsign:</b> {callsign}<br>
                  <b>IMO:</b> {imo}<br>
                  <b>Type:</b> {ship_type}<br>
                  <b>Length:</b> {length} m<br>
                  <b>Width:</b> {width} m
                </div>
                {{% endmacro %}}
                """
                macro = MacroElement()
                macro._template = Template(legend_html)
                m.get_root().add_child(macro)
        except Exception:
            # don't fail map creation due to legend issues
            pass

    if out_html is not None:
        m.save(out_html)

    return m
