from datetime import date, timedelta
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile


BASE_AIS_URL = "http://aisdata.ais.dk"  # Base URL for AIS data downloads

SEP_DATE1 = date.fromisoformat("2024-03-01") # data are saved monthly before this date
SEP_DATE2 = date.fromisoformat("2025-02-26") # data are saved with year/ before this date


def check_date_isdownloaded(day: date, dest_dir: Path) -> bool:
    """
    Check if the AIS data for the given date is already downloaded in dest_dir.

    ### Parameters:
        day (datetime.date): The date to check.
        dest_dir (pathlib.Path): The destination directory to check for existing data.

    ### Returns:
        bool: True if data for the date is already downloaded, False otherwise.
    """
    tag = day.strftime("%Y-%m") if day < SEP_DATE1 else day.strftime("%Y-%m-%d")
    return next(dest_dir.rglob(f"*{tag}*"), None) is not None


def get_work_dates(start: str, end: str, dest_dir: Path, filter: bool=True) -> list[date]:
    """
    Build and return the list of "anchor" dates to download for the given date range.

    ### Behavior:
        - For dates before SEP_DATE1, one anchor per month (the first day of the month) is returned.
        - For dates on/after SEP_DATE1, one anchor per day is returned.
        - The function creates out_dir if it doesn't exist and filters out anchors whose tag
        is already present anywhere under out_dir. Monthly tags are "YYYY-MM", daily tags
        are "YYYY-MM-DD".

    ### Parameters:
        start (str): inclusive start date in ISO format "YYYY-MM-DD".
        end (str): inclusive end date in ISO format "YYYY-MM-DD".
        out_dir (Path): destination directory used to check for already-downloaded data.
        filter (bool): if True, filter out already-downloaded dates; if False, return all dates.

    ### Returns:
        filtered_dates (list[date]): list of date objects representing the anchors to download.
    """

    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    # --- Build the schedule of download string dates ---
    work_dates = []

    def month_starts(d1: date, d2: date):
        """Yield the first day of each month between d1 and d2 (inclusive by month)."""
        y, m = d1.year, d1.month
        cur = date(y, m, 1)
        end_month = date(d2.year, d2.month, 1)
        while cur <= end_month:
            yield cur
            if m == 12:
                y += 1; m = 1
            else:
                m += 1
            cur = date(y, m, 1)

    # Monthly section: if range intersects anything between [start, SEP_DATE1]
    if start_date < SEP_DATE1:
        monthly_start = start_date
        monthly_end   = min(end_date, SEP_DATE1 - timedelta(days=1))
        for d in month_starts(monthly_start, monthly_end):
            work_dates.append(d)  # one entry per month

    # Daily section: if range intersects anything between [SEP_DATE1, end]
    if SEP_DATE1 <= end_date:
        daily_start = max(start_date, SEP_DATE1)
        d = daily_start
        while d <= end_date:
            work_dates.append(d)
            d += timedelta(days=1)

    # --- Filter out already-downloaded dates if requested ---
    filtered_dates = []
    if not filter:
        filtered_dates = work_dates
    else:
        # Remove dates already present in dest_dir (either zip files or extracted files/dirs)
        for day in work_dates:
            if not check_date_isdownloaded(day, dest_dir):
                filtered_dates.append(day)
        if not filtered_dates:
            print("All requested data are already present in the destination directory")

    return filtered_dates



def download_one_ais_data(day: date, dest_dir: Path) -> Path:
    """
    Download and extract AIS (Automatic Identification System) data for a single date.

    Constructs the correct download URL according to the historical URL scheme, downloads
    the ZIP archive with a byte-level tqdm progress bar, extracts all archive members
    into dest_dir with a member-count tqdm progress bar, then removes the downloaded ZIP.
    
    ### Behavior:
        - Check if the file is already downloaded, if so, skip download and extraction.
        - Build the download URL based on the date
        - Download with progress bar
        - The ZIP file is written to dest_dir folder
        - Extracted into dest_dir
        - Deleted zip file after extraction

    ### Parameters:
        day (datetime.date): Date to download (monthly or daily archive depending on thresholds).
        dest_dir (pathlib.Path): Destination directory where the ZIP will be saved and extracted.
            This directory must exist and be writable.

    ### Returns:
        csv_path (pathlib.Path): Path to the extracted CSV file.
    """

    # ---- Check if the file is already downloaded ----
    if day < SEP_DATE1:
        tag = f"{day:%Y-%m}"
    else:
        tag = f"{day:%Y-%m-%d}"

    #csv_path = Path(f"{dest_dir}/aisdk-{tag}.csv")

    csv_path = dest_dir / Path(f"aisdk-{tag}.csv")


    if check_date_isdownloaded(day, dest_dir):
        print(f"Skipping {tag} download: already present in {dest_dir} folder")
    else:
        print(f"Starting download and extraction for {tag}")

        # ---- Build the download URL based on the date ----
        if day < SEP_DATE2:
            # daily with year folder: .../{YYYY}/aisdk-{YYYY-MM-DD}.zip
            url = f"{BASE_AIS_URL}/{day:%Y}/aisdk-{tag}.zip"
        else:
            # daily without year folder: .../aisdk-{YYYY-MM-DD}.zip
            url = f"{BASE_AIS_URL}/aisdk-{tag}.zip"

        zip_path = dest_dir / Path(url).name

        # ---- Download with progress bar ----
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1 MB

            with open(zip_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {tag} zip file",
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
                        pbar.refresh()


        # ---- Unzip with progress bar ----
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(total=len(members), desc=f"Unzipping into {dest_dir} folder ") as pbar:
                for m in members:
                    zf.extract(m, path=dest_dir)
                    pbar.update(1)
                    csv_path = dest_dir / Path(m.filename)  # to avoid linting warning


        # ---- Delete the zip file after extraction ----
        zip_path.unlink()
        print(f"Completed download and extraction for {tag}")


    # ---- Return csv path ----
    return csv_path



def download_multiple_ais_data(start: str, end: str, dest_dir: Path) -> list[Path]:
    """
    Download AIS archives for a date range, extract them into dest_dir, and remove the downloaded zip files.

    ### Behavior:
        - Ensures dest_dir exists (creates it if necessary).
        - Uses get_work_dates(...) to determine which monthly/daily anchors to download.
        - Skips anchors whose tag (YYYY-MM or YYYY-MM-DD) already appears anywhere under dest_dir.
        - For each remaining anchor calls download_ais_data_one_day(...) to download, extract, and delete the zip.

    ### Parameters
        start (str): Inclusive start date in ISO format "YYYY-MM-DD".
        end (str): Inclusive end date in ISO format "YYYY-MM-DD".
        dest_dir (pathlib.Path): Destination directory where archives will be saved and extracted.

    ### Returns
        csv_paths (list[pathlib.Path]): List of paths to the extracted CSV files.
    """

    # --- Build the schedule of download string dates ---
    work_dates = get_work_dates(start, end, dest_dir)

    csv_paths = []

    # --- Iterate with tqdm and download, unzip and delete ---
    for day in tqdm(work_dates, desc="Processing download, unzip and delete", unit="file"):
        csv_path = download_one_ais_data(day, dest_dir)
        csv_paths.append(csv_path)

    return csv_paths