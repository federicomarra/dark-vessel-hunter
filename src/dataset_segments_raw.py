import torch
from torch.utils.data import Dataset
import pyarrow.dataset as ds
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional, Sequence, List, Tuple, Dict


class AISSegmentRaw(Dataset):
    """
    Dataset that scans a partitioned AIS Parquet dataset and returns
    fixed-length windows (chunks) of length `window_size`.

    Each (MMSI, Date, Segment) is a "segment".
    From each segment, we build a time-ordered sequence of feature vectors
    and then cut it into non-overlapping windows of length `window_size`.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int = 30,
        min_points: int = 30,
        allowed_dates: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            root_dir: Root folder of the Parquet dataset (e.g. 'ais-data/parquet').
            window_size: Number of timesteps per sample (e.g. 30).
            min_points: Minimum number of points required in a raw segment
                        to consider it (before windowing).
            allowed_dates: Optional list of date strings 'YYYY-MM-DD'.
                           If provided, only segments whose 'Date=...' matches
                           one of these values will be used.
        """
        super().__init__()
        self.root = Path(root_dir)
        self.window_size = int(window_size)
        self.min_points = int(min_points)
        self.allowed_dates = set(allowed_dates) if allowed_dates is not None else None

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root directory {self.root} not found.")

        # First pass: read all segments and collect raw feature arrays
        # plus ship type strings for global encoding.
        raw_segments: List[Dict[str, np.ndarray]] = []
        ship_type_values: List[str] = []

        for mmsi_dir in sorted(self.root.glob("MMSI=*")):
            mmsi = mmsi_dir.name.split("=", 1)[1]

            for date_dir in sorted(mmsi_dir.glob("Date=*")):
                date_str = date_dir.name.split("=", 1)[1]

                if self.allowed_dates is not None and date_str not in self.allowed_dates:
                    continue

                for seg_dir in sorted(date_dir.glob("Segment=*")):
                    seg_str = seg_dir.name.split("=", 1)[1]
                    try:
                        seg_id = int(seg_str)
                    except ValueError:
                        # Skip weird folder names
                        continue

                    dset = ds.dataset(str(seg_dir), format="parquet")
                    pdf = dset.to_table().to_pandas()

                    segment_data = self._build_segment_features(
                        pdf=pdf,
                        min_points=self.min_points,
                    )
                    if segment_data is None:
                        continue

                    segment_data["mmsi"] = mmsi
                    segment_data["date"] = date_str
                    segment_data["segment_id"] = seg_id
                    raw_segments.append(segment_data)

                    ship_type_values.append(segment_data["ship_type_str"])

        if not raw_segments:
            raise RuntimeError(f"No valid segments found in {self.root}.")

        # Build a global encoding for ship types (string -> integer index)
        unique_ship_types = sorted({s for s in ship_type_values})
        self.ship_type_to_idx = {name: idx for idx, name in enumerate(unique_ship_types)}
        self.idx_to_ship_type = {idx: name for name, idx in self.ship_type_to_idx.items()}

        # Second pass: convert all raw segments into (T, F) feature tensors
        # and collect them for global normalization.
        segment_tensors: List[torch.Tensor] = []
        self.segment_meta: List[Dict] = []

        for seg in raw_segments:
            feats = self._segment_to_feature_matrix(seg)
            if feats.shape[0] < self.window_size:
                # Too short to generate at least one window
                continue

            segment_tensors.append(torch.from_numpy(feats))  # (T, F)
            self.segment_meta.append(
                {
                    "mmsi": seg["mmsi"],
                    "date": seg["date"],
                    "segment_id": seg["segment_id"],
                    "length": feats.shape[0],
                }
            )

        if not segment_tensors:
            raise RuntimeError(
                f"All segments shorter than window_size={self.window_size} in {self.root}."
            )

        # Compute global mean/std across all segments for normalization
        all_feat = torch.cat(segment_tensors, dim=0)  # (sum_T, F)
        self.mean = all_feat.mean(dim=0)             # (F,)
        self.std = all_feat.std(dim=0).clamp_min(1e-6)

        # Build the list of windows (segment index, start index)
        windows: List[Tuple[int, int]] = []
        for s_idx, seq in enumerate(segment_tensors):
            T = seq.size(0)
            step = self.window_size  # non-overlapping windows
            for start in range(0, T - self.window_size + 1, step):
                windows.append((s_idx, start))

        self.segment_tensors = segment_tensors
        self.windows = windows
        self.num_features = segment_tensors[0].shape[-1]

        print(
            f"[AISSegmentRaw] root={self.root} | "
            f"segments={len(self.segment_tensors)} | "
            f"windows={len(self.windows)} | "
            f"features={self.num_features}"
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _build_segment_features(
        pdf: pd.DataFrame,
        min_points: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Given a pandas DataFrame for a single (MMSI, Date, Segment),
        build the raw per-timestep features before encoding and normalization.

        Returns a dict with numpy arrays if the segment is valid,
        or None if the segment is too short / malformed.

        Required columns:
            - 'Timestamp'
            - 'Latitude'
            - 'Longitude'
            - 'SOG'
            - 'COG'
        Optional columns:
            - 'Ship type'
        """
        required = ["Timestamp", "Latitude", "Longitude", "SOG", "COG"]
        missing = [c for c in required if c not in pdf.columns]
        if missing:
            # Segment does not have the required fields
            return None

        # Sort by time (just in case)
        pdf = pdf.sort_values("Timestamp")
        ts = pd.to_datetime(pdf["Timestamp"])

        # Convert numeric fields
        for col in ["Latitude", "Longitude", "SOG", "COG"]:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

        pdf = pdf.dropna(subset=["Latitude", "Longitude", "SOG", "COG"])
        if len(pdf) < min_points:
            return None

        # Recompute timestamps after drop
        ts = pd.to_datetime(pdf["Timestamp"])

        lat = pdf["Latitude"].to_numpy(np.float32)
        lon = pdf["Longitude"].to_numpy(np.float32)
        sog = pdf["SOG"].to_numpy(np.float32)
        cog_deg = pdf["COG"].to_numpy(np.float32)

        # Filter out non-finite values
        mask_valid = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(sog)
            & np.isfinite(cog_deg)
        )
        lat, lon, sog, cog_deg, ts = (
            lat[mask_valid],
            lon[mask_valid],
            sog[mask_valid],
            cog_deg[mask_valid],
            ts[mask_valid],
        )

        if lat.size < min_points:
            return None

        # Delta time in minutes between consecutive messages
        delta_t = ts.diff().dt.total_seconds().fillna(0.0).to_numpy(np.float32) / 60.0

        # Hour of day and day of week (for circular features)
        hour = ts.dt.hour.to_numpy(np.float32) + ts.dt.minute.to_numpy(np.float32) / 60.0
        dow = ts.dt.dayofweek.to_numpy(np.float32)  # 0=Monday

        hour_rad = 2.0 * np.pi * hour / 24.0
        dow_rad = 2.0 * np.pi * dow / 7.0

        hour_sin = np.sin(hour_rad).astype(np.float32)
        hour_cos = np.cos(hour_rad).astype(np.float32)
        dow_sin = np.sin(dow_rad).astype(np.float32)
        dow_cos = np.cos(dow_rad).astype(np.float32)

        # Ship type: take the first non-null value; if none, mark as 'UNKNOWN'
        if "Ship type" in pdf.columns:
            ship_col = pdf["Ship type"].astype(str)
            valid_ship = ship_col[ship_col.notna() & (ship_col != "nan")]
            ship_type_str = valid_ship.iloc[0] if not valid_ship.empty else "UNKNOWN"
        else:
            ship_type_str = "UNKNOWN"

        # Distance to cable: placeholder (will be computed later if available)
        # For now we set everything to zero.
        dist_to_cable = np.zeros_like(lat, dtype=np.float32)

        return {
            "lat": lat,
            "lon": lon,
            "sog": sog,
            "cog_deg": cog_deg,
            "delta_t": delta_t,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "dist_to_cable": dist_to_cable,
            "ship_type_str": ship_type_str,
        }

    def _segment_to_feature_matrix(self, seg: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert a raw segment dict into a (T, F) feature matrix.

        Features per timestep:
            - lat
            - lon
            - sog
            - sin(cog)
            - cos(cog)
            - delta_t
            - ship_type_idx (integer, as float)
            - dist_to_cable
            - sin(hour_of_day)
            - cos(hour_of_day)
            - sin(day_of_week)
            - cos(day_of_week)
        """
        lat = seg["lat"]
        lon = seg["lon"]
        sog = seg["sog"]
        cog_deg = seg["cog_deg"]
        delta_t = seg["delta_t"]
        hour_sin = seg["hour_sin"]
        hour_cos = seg["hour_cos"]
        dow_sin = seg["dow_sin"]
        dow_cos = seg["dow_cos"]
        dist_to_cable = seg["dist_to_cable"]

        # Convert COG in degrees to sin/Cos
        cog_rad = np.deg2rad(cog_deg.astype(np.float32))
        cog_sin = np.sin(cog_rad).astype(np.float32)
        cog_cos = np.cos(cog_rad).astype(np.float32)

        T = lat.shape[0]

        # Ship type index (global encoding)
        ship_name = seg["ship_type_str"]
        ship_idx = self.ship_type_to_idx.get(ship_name, 0)
        ship_feat = np.full(T, ship_idx, dtype=np.float32)

        # Stack all features along the last dimension -> (T, F)
        feats = np.stack(
            [
                lat,
                lon,
                sog,
                cog_sin,
                cog_cos,
                delta_t,
                ship_feat,
                dist_to_cable,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
            ],
            axis=-1,
        ).astype(np.float32)

        return feats

    # ---------------------------------------------------------------------
    # Standard Dataset API
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with key "x":
                - x: Tensor of shape (window_size, num_features), normalized.
        """
        seg_idx, start = self.windows[idx]
        seq = self.segment_tensors[seg_idx][start : start + self.window_size]  # (T, F)

        # Normalize using global mean and std
        # x = (seq - self.mean) / self.std
        x = (seq.to(self.mean.device) - self.mean) / self.std

        return {"x": x.float()}
