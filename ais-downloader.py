# If running in a fresh notebook, uncomment this:
# !pip -q install tqdm requests

from datetime import date, timedelta
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile


# Set these
START_DATE = "2025-02-04"
END_DATE   = "2025-02-11"

FOLDER_NAME = "ais-data"



# Do not touch from under here
BASE_AIS_URL = "http://aisdata.ais.dk"
DEST_DIR = Path(FOLDER_NAME)                        # change if you like
DEST_DIR.mkdir(parents=True, exist_ok=True)        # create folders if missing


start = date.fromisoformat(START_DATE)
end   = date.fromisoformat(END_DATE)
separation1 = date.fromisoformat("2024-03-01") # data are saved monthly before this date
separation2 = date.fromisoformat("2025-02-26") # data are saved with year/ before this date

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

# Monthly section: if range intersects anything between [start, separation1]
if start < separation1:
    monthly_start = start
    monthly_end   = min(end, separation1 - timedelta(days=1))
    for d in month_starts(monthly_start, monthly_end):
        work_dates.append(d)  # one entry per month

# Daily section: if range intersects anything between [separation1, end]
if separation1 <= end:
    daily_start = max(start, separation1)
    d = daily_start
    while d <= end:
        work_dates.append(d)
        d += timedelta(days=1)

# Remove dates already present in DEST_DIR (either zip files or extracted files/dirs)
filtered_dates = []
for d in work_dates:
    # monthly tags are YYYY-MM, daily tags are YYYY-MM-DD
    tag = d.strftime("%Y-%m") if d < separation1 else d.strftime("%Y-%m-%d")
    # if any file/dir under DEST_DIR contains the tag, consider it already downloaded
    if next(DEST_DIR.rglob(f"*{tag}*"), None) is None:
        filtered_dates.append(d)

work_dates = filtered_dates



# --- iterate with tqdm and build the correct URL for each anchor date ---
for d in tqdm(work_dates, desc="Processing download, unzip and delete", unit="file"):
    if d < separation1:
        # monthly file: .../{YYYY}/aisdk-{YYYY-MM}.zip
        url = f"{BASE_AIS_URL}/{d:%Y}/aisdk-{d:%Y-%m}.zip"
    elif d < separation2:
        # daily with year folder: .../{YYYY}/aisdk-{YYYY-MM-DD}.zip
        url = f"{BASE_AIS_URL}/{d:%Y}/aisdk-{d:%Y-%m-%d}.zip"
    else:
        # daily without year folder: .../aisdk-{YYYY-MM-DD}.zip
        url = f"{BASE_AIS_URL}/aisdk-{d:%Y-%m-%d}.zip"

    zip_path = DEST_DIR / Path(url).name

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
        with tqdm(total=len(members), desc=f"Unzipping to {DEST_DIR}") as pbar:
            for m in members:
                zf.extract(m, path=DEST_DIR)
                pbar.update(1)


    # ---- Delete the zip file after extraction ----
    zip_path.unlink()

