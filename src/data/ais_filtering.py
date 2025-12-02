import pandas as pd
from pathlib import Path

from typing import Sequence, Optional
from collections.abc import Sequence
from shapely.geometry import Polygon, MultiPolygon
from shapely import contains_xy

def filter_ais_df(
    df: pd.DataFrame,
    polygon_coords: Sequence[tuple[float, float]],
    allowed_mobile_types: Optional[Sequence[str]] = ("Class A", "Class B"),
    apply_polygon_filter: bool = True,
    remove_zero_sog_vessels: bool = True,
    output_sog_in_ms: bool = True,         
    sog_min_knots: float | None = 0.5,      
    sog_max_knots: float | None = 35.0,     
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
    4) Optional polygon filtering using Shapely (lon, lat)
    5) Optional removal of AIS points inside port polygons
    6) COG sanity filter (keep only 0–360 deg)
    7) SOG sanity filter (remove unrealistic speeds; assumes input SOG in knots)
    8) Optional conversion of SOG from knots to m/s (if output_sog_in_ms=True)
    9) Optional removal of ships with >90% zero SOG
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
    # 4) Polygon filtering (vectorized)
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
    # 5) Remove AIS points inside port polygons (from port_locodes.csv)
    # ------------------------------------------------------------------
    if (
        exclude_ports
        and port_locodes_path is not None
        and apply_polygon_filter
        and polygon_coords is not None
    ):
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
    # 6) COG sanity filter (0–360 degrees)
    # ------------------------------------------------------------------
    if "COG" in df.columns:
        df["COG"] = pd.to_numeric(df["COG"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["COG"])
        df = df[(df["COG"] >= 0.0) & (df["COG"] <= 360.0)]

        if verbose:
            print(
                f" [filter_ais_df] COG sanity: {len(df):,} rows "
                f"(removed {before - len(df):,}) with range [0, 360] deg"
            )
    elif verbose:
        print(" [filter_ais_df] Warning: 'COG' column not found, skipping COG filter.")

    # ------------------------------------------------------------------
    # 7) SOG sanity filter (in knots)
    # ------------------------------------------------------------------
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df = df.dropna(subset=["SOG"])

    if sog_min_knots is not None or sog_max_knots is not None:
        before = len(df)
        mask = pd.Series(True, index=df.index)

        if sog_min_knots is not None:
            mask &= df["SOG"] >= sog_min_knots
        if sog_max_knots is not None:
            mask &= df["SOG"] <= sog_max_knots

        df = df[mask]

        if verbose:
            print(
                f" [filter_ais_df] SOG sanity: {len(df):,} rows "
                f"(removed {before - len(df):,}) "
                f"with range [{sog_min_knots}, {sog_max_knots}] knots"
            )

    # ------------------------------------------------------------------
    # 8) SOG conversion to m/s (optional)
    # ------------------------------------------------------------------
    if output_sog_in_ms:
        df["SOG"] = df["SOG"].astype(float) * 0.514444

    # ------------------------------------------------------------------
    # 9) Remove vessels with >90% zero SOG
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
        unit = "m/s" if output_sog_in_ms else "knots"
        print(
            f" [filter_ais_df] Final: {len(df):,} rows, "
            f"{df['MMSI'].nunique():,} unique vessels "
            f"(SOG in {unit})"
        )
        
    # --- Add Date ---
    ddate = df["Timestamp"].dt.strftime("%Y-%m-%d")
    df["Date"] = ddate.astype(str)
    
    df["TrackID"] = df["MMSI"].astype(str) + "_" + ddate.astype(str)
    

    return df

