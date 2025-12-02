from pathlib import Path
from typing import Union
import shutil

import pandas as pd
import pyarrow
import pyarrow.parquet


def save_by_mmsi(
    df: pd.DataFrame,
    verbose: bool = False,
    output_folder: Union[Path, str] = "ais_data/parquet",
    timestamp_col: str = "Timestamp",
    date_col: str = "Date",
) -> Path:
    """
    Write AIS data to a partitioned Parquet dataset, partitioned by
    MMSI and Date (YYYY-MM-DD).

    The output directory is **always** `output_folder` (default "ais_data_parquet").
    If it does not exist, it will be created.

    IMPORTANT
    ---------
    This function is *overwrite-safe* for the partitions present in `df`.
    For each unique (MMSI, Date) combination in `df`, the existing partition
    directory is removed before writing new data. This avoids accumulating
    multiple parquet files for the same MMSI+Date when rerunning the pipeline
    for the same day/file.

    Expected columns
    ----------------
    df must contain:
    - "MMSI"           (string-like)
    - "Timestamp"      (datetime-like) OR an existing "Date" column
      If "Date" is missing, it will be created from `Timestamp` as YYYY-MM-DD.

    Partition layout
    ----------------
    ais_data_parquet/
        MMSI=123456789/
            Date=2025-11-05/part-*.parquet
        MMSI=987654321/
            Date=2025-11-06/part-*.parquet

    Parameters
    ----------
    df : pd.DataFrame
        AIS DataFrame containing at least "MMSI" and either "Timestamp" or "Date".
    verbose : bool, optional
        If True, print the output path and some info.
    output_folder : Union[Path, str], optional
        Root folder for the parquet dataset.
    timestamp_col : str, optional
        Name of timestamp column used to derive Date if needed.
    date_col : str, optional
        Name of the Date column (default "Date").

    Returns
    -------
    Path
        Path to the root parquet dataset folder.
    """
    df = df.copy()

    # --- Basic checks ---
    if "MMSI" not in df.columns:
        raise KeyError("save_by_mmsi: required column 'MMSI' missing")

    # Ensure MMSI is string
    df["MMSI"] = df["MMSI"].astype(str)

    # --- Ensure Date column exists ---
    if date_col not in df.columns:
        if timestamp_col not in df.columns:
            raise KeyError(
                f"save_by_mmsi: neither '{date_col}' nor '{timestamp_col}' found in df"
            )

        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            # Try to convert if it's not datetime yet
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")

        df[date_col] = df[timestamp_col].dt.strftime("%Y-%m-%d")

    # Make sure Date is string
    df[date_col] = df[date_col].astype(str)

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Remove existing folders for (MMSI, Date) in df (overwrite-safe)
    # ------------------------------------------------------------------
    partitions = df[["MMSI", date_col]].drop_duplicates()

    for _, row in partitions.iterrows():
        mmsi_val = row["MMSI"]
        date_val = row[date_col]

        part_dir = out_path / f"MMSI={mmsi_val}" / f"{date_col}={date_val}"

        if part_dir.exists():
            shutil.rmtree(part_dir)
            
    if verbose:
        print(f" [save_by_mmsi] Removed existing partitions for {len(partitions)} (MMSI, Date) combinations.")

    # ------------------------------------------------------------------
    # Write new dataset (append is fine now that partitions are cleaned)
    # ------------------------------------------------------------------
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=["MMSI", date_col],
    )

    if verbose:
        print(f" [save_by_mmsi] Parquet dataset written/appended at: {out_path.resolve()}")

    return out_path