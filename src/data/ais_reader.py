from __future__ import annotations
from pathlib import Path
from typing import Sequence, Union

import duckdb
import pandas as pd
from shapely.geometry import Polygon
from shapely import contains_xy




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
    FROM read_csv_auto('{csv_path}', AUTO_DETECT=TRUE, ignore_errors=TRUE)
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



def read_raw_csv_with_filters(
    csv_name: Union[Path, str],
    bbox: Sequence[float],
    time_start: Union[str, pd.Timestamp, None],
    time_end: Union[str, pd.Timestamp, None],
    csv_root: Union[str, Path] = "ais-data/csv",
    timestamp_format: str = "%d/%m/%Y %H:%M:%S",
    polygon_coords: Sequence[tuple[float, float]] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read a raw AIS CSV using DuckDB, applying:

    - Spatial filtering via a bounding box (lat/lon)
    - Optional time filtering via `time_start` / `time_end`
    - Optional polygon mask (lon, lat) using Shapely for precise AOI

    Parameters
    ----------
    csv_name : str or Path
        Name of the CSV file, e.g. 'aisdk-2025-05-21.csv'.
    bbox : Sequence[float]
        Bounding box as [lat_max, lon_min, lat_min, lon_max].
    time_start : str, Timestamp, or None
        Start of time window (inclusive). If None, no time filter is applied.
    time_end : str, Timestamp, or None
        End of time window (inclusive). If None, no time filter is applied.
    csv_root : str or Path
        Root directory where CSV files are stored.
    timestamp_format : str
        Format of the timestamp string in the CSV.
    polygon_coords : Sequence[(lon, lat)] or None
        Optional polygon vertices for more precise AOI filtering.
    verbose : bool
        If True, prints a short summary of the result.

    Returns
    -------
    pd.DataFrame
        Filtered AIS records with at least:
        ['Latitude', 'Longitude', 'Timestamp', 'MMSI', 'SOG'].
    """

    # ---------------------------------------------------------------------
    # Resolve paths and basic parameters
    # ---------------------------------------------------------------------
    csv_path = Path(csv_root) / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"No AIS CSV at: {csv_path}")

    lat_max, lon_min, lat_min, lon_max = bbox

    # Optional time range
    use_time_filter = time_start is not None and time_end is not None
    if use_time_filter:
        ts_start = pd.Timestamp(time_start)
        ts_end = pd.Timestamp(time_end)

    # ---------------------------------------------------------------------
    # Build DuckDB SQL query
    # ---------------------------------------------------------------------
    base_query = f"""
        SELECT *
        FROM read_csv_auto('{csv_path.as_posix()}', AUTO_DETECT=TRUE, ignore_errors=TRUE)
        WHERE Latitude BETWEEN {lat_min} AND {lat_max}
          AND Longitude BETWEEN {lon_min} AND {lon_max}
    """

    if use_time_filter:
        # Use CAST to VARCHAR before strptime in case the column is already TIMESTAMP
        time_filter = f"""
          AND COALESCE(
                strptime(CAST("# Timestamp" AS VARCHAR), '{timestamp_format}'),
                "# Timestamp"
              ) BETWEEN '{ts_start}' AND '{ts_end}'
        """
        query = base_query + time_filter + ";"
    else:
        query = base_query + ";"

    # ---------------------------------------------------------------------
    # Execute query and normalize schema
    # ---------------------------------------------------------------------
    df = duckdb.query(query).to_df()

    # Normalize timestamp column name
    if "# Timestamp" in df.columns and "Timestamp" not in df.columns:
        df = df.rename(columns={"# Timestamp": "Timestamp"})

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=timestamp_format, errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    # Ensure required columns are present
    required_columns = ["Latitude", "Longitude", "Timestamp", "MMSI", "SOG"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    # Normalize MMSI type
    df["MMSI"] = df["MMSI"].astype(str)

    # ---------------------------------------------------------------------
    # Optional polygon filter (lon, lat)
    # ---------------------------------------------------------------------
    if polygon_coords is not None:
        poly = Polygon(polygon_coords)
        lons = df["Longitude"].to_numpy()
        lats = df["Latitude"].to_numpy()
        mask = contains_xy(poly, lons, lats)
        df = df[mask].copy()

    # ---------------------------------------------------------------------
    # Verbose summary
    # ---------------------------------------------------------------------
    if verbose:
        if use_time_filter:
            print(
                f"[read_raw_csv_with_filters] {len(df):,} rows, {df['MMSI'].nunique():,} vessels; "
                f"time between {ts_start} and {ts_end}"
            )
        else:
            print(
                f"[read_raw_csv_with_filters] {len(df):,} rows, {df['MMSI'].nunique():,} vessels; "
                "no time filter applied"
            )

    return df
