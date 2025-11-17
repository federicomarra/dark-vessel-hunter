import torch
from torch.utils.data import Dataset
# pip install pyarrow
import pyarrow.dataset as ds
import numpy as np
from pathlib import Path
import pandas as pd

def clean_and_to_seq(pdf, min_length: int):
    """Pulisce il DataFrame e restituisce (ok, tensor[T,5]) oppure (False, None)."""

    # Ordina per timestamp se presente
    if "Timestamp" in pdf.columns:
        pdf = pdf.sort_values("Timestamp")

    # Converte in numerico con coercizione
    for col in ["Latitude", "Longitude", "SOG", "COG"]:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    # Drop righe che hanno NaN in una delle colonne chiave
    pdf = pdf.dropna(subset=["Latitude", "Longitude", "SOG", "COG"])
    if len(pdf) < min_length:
        return False, None

    # Numpy arrays
    lat = pdf["Latitude"].to_numpy(np.float32)
    lon = pdf["Longitude"].to_numpy(np.float32)
    sog = pdf["SOG"].to_numpy(np.float32)
    cog = pdf["COG"].to_numpy(np.float32)

    # Filtra eventuali inf
    mask_valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(sog) & np.isfinite(cog)
    lat, lon, sog, cog = lat[mask_valid], lon[mask_valid], sog[mask_valid], cog[mask_valid]

    if lat.size < min_length:
        return False, None

    cog_rad = np.deg2rad(cog)
    x = np.stack([lat, lon, sog, np.cos(cog_rad), np.sin(cog_rad)], axis=-1)  # (T,5)
    return True, torch.from_numpy(x)

class AISSegmentRaw(Dataset):
    """
    Dataset che scansiona cartelle:
      <root_day>/MMSI=.../Segment=.../*.parquet

    e tiene SOLO i segmenti che, dopo pulizia, hanno almeno `min_length` punti validi.
    """
    def __init__(self, root_dir: str, min_length: int = 1):
        self.min_length = int(min_length)
        root = Path(root_dir)

        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory {root} not found.")

        paths = []
        data = []

        # 1) Scansiona tutte le cartelle MMSI=... / Segment=...
        for mdir in sorted(root.glob("MMSI=*")):   # <-- IMPORTANTE: "MMSI=*"
            mmsi = mdir.name.split("=", 1)[1]
            for sdir in sorted(mdir.glob("Segment=*")):
                seg_str = sdir.name.split("=", 1)[1]
                try:
                    seg = int(seg_str)
                except ValueError:
                    continue
                # legge il/i parquet del segmento
                dset = ds.dataset(str(sdir), format="parquet")
                pdf = dset.to_table().to_pandas()
                ok, x = clean_and_to_seq(pdf, min_length=self.min_length)
                if ok:
                    paths.append((mmsi, seg, sdir))
                    data.append(x)

        if not paths:
            raise RuntimeError(f"No valid MMSI/Segment sequences found in {root_dir}.")

        # salva paths e tensori già pronti
        self.paths = paths
        self.data = data
        print(f"[AISSegmentRaw] Loaded {len(self.paths)} valid segments from {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        # abbiamo già i tensori puliti in self.data
        return {"x": self.data[index]} 