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
