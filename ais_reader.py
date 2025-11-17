from pathlib import Path
from typing import Sequence, Union

import duckdb
import pandas as pd

def read_single_ais_df(
    csv_path: Union[Path, str],
    bbox: Sequence[float],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read a single AIS CSV into a DataFrame using DuckDB and apply basic cleaning.

    Operations:
    - Spatial bbox filter in the SQL
    - Rename & parse Timestamp
    - Drop rows with invalid timestamps
    - Check required columns exist
    - Ensure MMSI is a string

    Parameters
    ----------
    csv_path : Path | str
        Path to the AIS CSV file.
    bbox : Sequence[float]
        Bounding box as [lat_max, lon_min, lat_min, lon_max].
    verbose : bool, optional
        If True, print basic info about the loaded data.

    Returns
    -------
    pd.DataFrame
        AIS data within the given bounding box, with basic cleaning applied.

    Examples
    --------
    >>> bbox = [57.58, 10.5, 57.12, 11.92]
    >>> df_raw = read_ais_df("ais-data/aisdk-2025-11-05.csv", bbox, verbose=True)
    """
    lat_max, lon_min, lat_min, lon_max = bbox

    query = f"""
    SELECT *
    FROM read_csv_auto('{csv_path}', AUTO_DETECT=TRUE)
    WHERE Latitude <= {lat_max}
      AND Latitude >= {lat_min}
      AND Longitude >= {lon_min}
      AND Longitude <= {lon_max}
    ;
    """

    df = duckdb.query(query).to_df()

    # Rename Timestamp column and parse to datetime
    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )

    # Drop rows where timestamp parsing failed
    df = df.dropna(subset=["Timestamp"])

    # Basic column checks
    required_columns = ["Latitude", "Longitude", "Timestamp", "MMSI", "SOG"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f" Required columns missing: {missing}")

    # Ensure MMSI is string for later processing
    df["MMSI"] = df["MMSI"].astype(str)

    if verbose:
        print(
            f" Read AIS data: {len(df):,} rows within bbox, "
            f" {df['MMSI'].nunique():,} unique vessels"
        )

    return df



from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.predicates import contains_xy


def filter_ais_df(
    df: pd.DataFrame,
    polygon_coords: Sequence[Sequence[tuple[float, float]]],
    allowed_mobile_types: Optional[Sequence[str]] = ("Class A", "Class B"),
    bbox: Optional[Sequence[float]] = None,
    apply_polygon_filter: bool = True,
    remove_zero_sog_vessels: bool = True,
    sog_in_knots: bool = True,
    port_polygons: Optional[Sequence[Polygon]] = None,
    exclude_port_areas: bool = False,
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
    5b) Optional exclusion of overlapping port polygons
    6) Convert SOG from knots to m/s (if sog_in_knots=True)
    7) Optional removal of ships with >90% zero SOG
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
        print(f" [filter_ais_df] Before filtering: {len(df):,} rows, {df['MMSI'].nunique():,} vessels")

    # Build main AOI polygon once (used for #5 and port overlap)
    main_polygon = Polygon(polygon_coords)

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
    if apply_polygon_filter and polygon_coords is not None:
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
    # 6) Exclude port polygons overlapping AOI
    # ------------------------------------------------------------------
    if exclude_port_areas and port_polygons is not None:
        # keep only ports overlapping the main polygon (AOI)
        overlapping_ports = [pp for pp in port_polygons if pp.intersects(main_polygon)]

        if overlapping_ports:
            lons = df["Longitude"].to_numpy()
            lats = df["Latitude"].to_numpy()

            mask_in_ports = np.zeros(len(df), dtype=bool)
            for pp in overlapping_ports:
                mask_in_ports |= contains_xy(pp, lons, lats)

            before = len(df)
            df = df[~mask_in_ports].copy()

            if verbose:
                print(
                    f" [filter_ais_df] Port-area exclusion: {len(df):,} rows "
                    f"(removed {before - len(df):,}), {df['MMSI'].nunique():,} vessels"
                )
        elif verbose:
            print(" [filter_ais_df] No port polygons intersect AOI, skipping port exclusion.")

    # ------------------------------------------------------------------
    # 6) SOG conversion to m/s
    # ------------------------------------------------------------------
    if sog_in_knots:
        df["SOG"] = df["SOG"].astype(float) * 0.514444

    # ------------------------------------------------------------------
    # 7) Remove vessels with >90% zero SOG
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
