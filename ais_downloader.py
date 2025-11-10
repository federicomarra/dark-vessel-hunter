from datetime import date, timedelta
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile as zipfile


BASE_AIS_URL = "http://aisdata.ais.dk"  # Base URL for AIS data downloads

SEP_DATE1 = date.fromisoformat("2024-03-01") # data are saved monthly before this date
SEP_DATE2 = date.fromisoformat("2025-02-26") # data are saved with year/ before this date


def check_date_isdownloaded(day: date, dest_dir: Path) -> bool:
    """
    Check if the AIS data for the given date is already downloaded in dest_dir.

    Parameters:
    - day (date): The date to check.
    - dest_dir (Path): The destination directory to check for existing data.

    Returns:
    - bool: True if data for the date is already downloaded, False otherwise.
    """
    tag = day.strftime("%Y-%m") if day < SEP_DATE1 else day.strftime("%Y-%m-%d")
    return next(dest_dir.rglob(f"*{tag}*"), None) is not None


def get_work_dates(start: str, end: str, dest_dir: Path, filter: bool=True) -> list[date]:
    """
    Build and return the list of "anchor" dates to download for the given date range.

    Behavior:
    - For dates before SEP_DATE1, one anchor per month (the first day of the month) is returned.
    - For dates on/after SEP_DATE1, one anchor per day is returned.
    - The function creates out_dir if it doesn't exist and filters out anchors whose tag
      is already present anywhere under out_dir. Monthly tags are "YYYY-MM", daily tags
      are "YYYY-MM-DD".

    Parameters:
    - start (str): inclusive start date in ISO format "YYYY-MM-DD".
    - end (str): inclusive end date in ISO format "YYYY-MM-DD".
    - out_dir (Path): destination directory used to check for already-downloaded data.
    - filter (bool): if True, filter out already-downloaded dates; if False, return all dates.

    Returns:
    - list[date]: list of date objects representing the anchors to download.

    Raises:
    - ValueError: if start or end are not valid ISO dates or if start > end (propagated
      from datetime.date.fromisoformat).
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

    filtered_dates = []
    if not filter:
        filtered_dates = work_dates
    else:
        # Remove dates already present in dest_dir (either zip files or extracted files/dirs)
        for day in work_dates:
            if not check_date_isdownloaded(day, dest_dir):
                filtered_dates.append(day)
        if not filtered_dates:
            print("All requested data are already present in the destination directory.")

    return filtered_dates



def download_ais_data_one_day(day: date, dest_dir: Path):
    """
    Download and extract AIS (Automatic Identification System) data for a single day.

    This function constructs the appropriate download URL for a given date according to
    the repository's historical URL scheme, downloads the ZIP archive with a
    console progress bar, extracts all files to the specified destination directory
    (with a separate extraction progress bar), and removes the downloaded ZIP file
    after successful extraction.

    URL selection logic (based on global constants SEP_DATE1, SEP_DATE2 and BASE_AIS_URL):
    - For dates d < SEP_DATE1: monthly archive under a year folder:
        {BASE_AIS_URL}/{YYYY}/aisdk-{YYYY-MM}.zip
    - For SEP_DATE1 <= d < SEP_DATE2: daily archive under a year folder:
        {BASE_AIS_URL}/{YYYY}/aisdk-{YYYY-MM-DD}.zip
    - For d >= SEP_DATE2: daily archive without year folder:
        {BASE_AIS_URL}/aisdk-{YYYY-MM-DD}.zip

    Parameters
    ----------
    d : datetime.date
        The date for which AIS data should be downloaded and extracted.
    dest_dir : pathlib.Path
        Destination directory where the ZIP file will be saved and its contents extracted.
        This directory MUST exist and be writable by the process. The name of the
        downloaded ZIP file is derived from the download URL and is saved as
        dest_dir / <zip_filename>.

    Behavior and side effects
    -------------------------
    - Performs an HTTP GET request (streamed) with a 60 second timeout to download the ZIP.
    - Displays a tqdm progress bar for download (bytes) and a second tqdm progress bar
      showing the number of archive members extracted.
    - Writes the ZIP file to disk as dest_dir/<zip_filename>, extracts all members
      into dest_dir, and deletes the ZIP file after extraction.
    - Uses a 1 MB chunk size when streaming the download.

    Exceptions
    ----------
    - requests.HTTPError (or requests.RequestException) if the HTTP request fails or the
      response status is not 200.
    - OSError (e.g., FileNotFoundError, PermissionError) if writing to disk or extracting
      files fails.
    - zipfile.BadZipFile if the downloaded file is not a valid ZIP archive.
    - Any other IO-related exceptions encountered during download/extraction.

    Notes
    -----
    - The function relies on the module-level constants SEP_DATE1, SEP_DATE2, and
      BASE_AIS_URL to determine the download URL format.
    - The function does not create dest_dir; ensure the directory exists beforehand.
    - Progress bars are printed to the console (tqdm); in non-interactive environments
      this will still emit progress information to standard output.
    - The ZIP file is removed after successful extraction; if extraction fails, the
      ZIP may remain on disk for debugging.

    Example
    -------

    download_ais_data_one_day(date(2025, 11, 10), Path("/ais-data"))
    """

    # ---- Check if the file is already downloaded ----
    tag = day.strftime("%Y-%m") if day < SEP_DATE1 else day.strftime("%Y-%m-%d")
    if check_date_isdownloaded(day, dest_dir):
        print(f"Skipping {tag} download: already present in {dest_dir} folder.")
        return

    # ---- Build the download URL based on the date ----
    if day < SEP_DATE1:
        # monthly file: .../{YYYY}/aisdk-{YYYY-MM}.zip
        url = f"{BASE_AIS_URL}/{day:%Y}/aisdk-{day:%Y-%m}.zip"
    elif day < SEP_DATE2:
        # daily with year folder: .../{YYYY}/aisdk-{YYYY-MM-DD}.zip
        url = f"{BASE_AIS_URL}/{day:%Y}/aisdk-{day:%Y-%m-%d}.zip"
    else:
        # daily without year folder: .../aisdk-{YYYY-MM-DD}.zip
        url = f"{BASE_AIS_URL}/aisdk-{day:%Y-%m-%d}.zip"

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
            desc=zip_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))


    # ---- Unzip with progress bar ----
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(total=len(members), desc=f"Unzipping to {dest_dir}") as pbar:
            for m in members:
                zf.extract(m, path=dest_dir)
                pbar.update(1)


    # ---- Delete the zip file after extraction ----
    zip_path.unlink()

    print(f"Completed download and extraction for {tag}.")



def download_ais_data_multiple_days(start: str, end: str, dest_dir: Path):
    """
    Download AIS archives for a date range, extract them into dest_dir, and remove the downloaded zip files.

    This function:
    - Ensures dest_dir exists (creates it if necessary).
    - Uses get_work_dates(...) to determine which monthly/daily anchors to download.
    - Skips anchors whose tag (YYYY-MM or YYYY-MM-DD) already appears anywhere under dest_dir.
    - For each remaining anchor calls download_ais_data_one_day(...) to download, extract, and delete the zip.

    Parameters
    ----------
    start : str
        Inclusive start date in ISO format "YYYY-MM-DD".
    end : str
        Inclusive end date in ISO format "YYYY-MM-DD".
    dest_dir : pathlib.Path
        Destination directory where archives will be saved and extracted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If start or end are not valid ISO dates, or start > end (propagated from date operations).
    requests.exceptions.RequestException
        For network-related errors (including non-2xx responses via raise_for_status).
    zipfile.BadZipFile
        If a downloaded file is not a valid zip archive.
    OSError
        For filesystem errors (creating/writing/deleting files).
    """


    # --- Build the schedule of download string dates ---
    work_dates = get_work_dates(start, end, dest_dir)


    # --- Iterate with tqdm and download, unzip and delete ---
    for d in tqdm(work_dates, desc="Processing download, unzip and delete", unit="file"):
        download_ais_data_one_day(d, dest_dir)

        