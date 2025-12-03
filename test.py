"""
======================================================
AIS Autoencoder – Training & Evaluation Utilities
======================================================

Structure:

- PART A: Dataset with meta + generic evaluation functions
- PART B: PHASE 1 – Training on train/val + best model selection
- PART C: PHASE 2 – Loading saved model + evaluation on test
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score)

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


import config as config_file
from src.utils import training_utils
from src.utils.main_test_inspection_utils import run_test_inspection_pipeline, create_map_for_mmsi, create_map_for_segments

from src.train.models.AE_simple import LSTMAutoencoderWithShipType as TrainModel
from src.train.models.loss_fn import sequence_loss_fn as sequence_loss_fn



# ======================================================
# PART A – Dataset with meta + generic evaluation functions
# ======================================================

class AISTrajectoryDataset(Dataset):
    """
    Dataset that wraps df_seq.

    Modes:
      - return_meta = False (default): used for TRAINING
          __getitem__ -> (x, ship_type_id)

      - return_meta = True: used for EVALUATION
          __getitem__ -> (x, ship_type_id, segment_nr, mmsi)
    """

    def __init__(self, df_seq: pd.DataFrame, return_meta: bool = False):
        self.df = df_seq.reset_index(drop=True)
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # AIS sequence
        seq = np.array(row["Sequence"], dtype=np.float32)  # (T, Features)
        x = torch.from_numpy(seq)                          # (T, Features)

        # Ship type (always)
        ship_type_id = int(row["ShipTypeID"])
        ship_type_id = torch.tensor(ship_type_id, dtype=torch.long)

        # Simple mode -> training
        if not self.return_meta:
            return x, ship_type_id

        # Mode with meta -> evaluation
        seg = row["Segment_nr"]
        segment_nr = int(seg)

        # MMSI (if missing or NaN -> -1)
        m = row["MMSI"]
        mmsi = int(m)

        return x, ship_type_id, segment_nr, mmsi



def load_and_split_train_val(
    train_df_path: str,
    val_size: float = 0.2,
    random_state: int = 5,
    split_by_mmsi: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the preprocessed DF from train_df_path and split it into df_seq_train / df_seq_val.

    If split_by_mmsi=True:
      - use GroupShuffleSplit so that no MMSI appears in both sets.

    If split_by_mmsi=False:
      - use a regular random train_test_split.
    """
    # Load full df + any custom parsing
    df_seq_all = training_utils.load_df_seq(train_df_path)

    if split_by_mmsi:
        # Drop any rows without MMSI (or you can handle them separately)
        df_seq_all = df_seq_all.dropna(subset=["MMSI"])
        groups = df_seq_all["MMSI"].values

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=random_state,
        )
        train_idx, val_idx = next(gss.split(df_seq_all, groups=groups))

        df_seq_train = df_seq_all.iloc[train_idx].reset_index(drop=True)
        df_seq_val   = df_seq_all.iloc[val_idx].reset_index(drop=True)

    else:
        df_seq_train, df_seq_val = train_test_split(
            df_seq_all,
            test_size=val_size,
            random_state=random_state,
        )
        df_seq_train = df_seq_train.reset_index(drop=True)
        df_seq_val   = df_seq_val.reset_index(drop=True)

    print("Train segments:", len(df_seq_train))
    print("Val   segments:", len(df_seq_val))
    print("Unique MMSI train:", df_seq_train["MMSI"].nunique())
    print("Unique MMSI val  :", df_seq_val["MMSI"].nunique())
    if split_by_mmsi:
        overlap = len(set(df_seq_train["MMSI"]) & set(df_seq_val["MMSI"]))
        print("Overlap MMSI     :", overlap)

    return df_seq_train, df_seq_val


def evaluate_model_on_df(
    model: torch.nn.Module,
    df_seq: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    set_name: str = "set",
) -> pd.DataFrame:
    """
    Compute the reconstruction error for each segment in df_seq.

    Returns a DataFrame with columns:
      - set
      - Segment_nr
      - ShipTypeID
      - MMSI (if available, otherwise None)
      - reconstruction_error
    """
    # HERE we use the single dataset, in meta mode:
    dataset = AISTrajectoryDataset(df_seq, return_meta=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x, ship_type_id, segment_nr, mmsi in loader:
            x = x.to(device)                       # (B, T, F)
            ship_type_id = ship_type_id.to(device) # (B,)

            recon, _ = model(x, ship_type_id)      # (B, T, F)
            seq_errors = sequence_loss_fn(recon, x)  # (B,)

            seq_errors = seq_errors.cpu().numpy()
            segment_nr = segment_nr.cpu().numpy()
            mmsi = mmsi.cpu().numpy()
            ship_type_id_np = ship_type_id.cpu().numpy()

            for e, seg, st, m in zip(seq_errors, segment_nr, ship_type_id_np, mmsi):
                records.append(
                    {
                        "set": set_name,
                        "Segment_nr": int(seg),
                        "ShipTypeID": int(st),
                        "MMSI": int(m) if m != -1 else None,
                        "reconstruction_error": float(e),
                    }
                )

    df_errors = pd.DataFrame(records)
    return df_errors



def summarize_reconstruction_errors(
    df_errors: pd.DataFrame,
    set_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Global statistics on the error:
      - count, mean, std, min, max, median, p90, p95, p99, (p999 if enough data)
    """
    if set_name is None:
        if "set" in df_errors.columns:
            set_name = str(df_errors["set"].iloc[0])
        else:
            set_name = "set"

    errs = df_errors["reconstruction_error"].values

    stats = {
        "set": set_name,
        "count": int(len(errs)),
        "mean": float(np.mean(errs)),
        "std": float(np.std(errs)),
        "min": float(np.min(errs)),
        "max": float(np.max(errs)),
        "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)),
        "p95": float(np.percentile(errs, 95)),
        "p99": float(np.percentile(errs, 99)),
    }

    if len(errs) >= 1000:
        stats["p999"] = float(np.percentile(errs, 99.9))

    return stats


def summarize_errors_by_shiptype(df_errors: pd.DataFrame) -> pd.DataFrame:
    """
    Statistics per ShipTypeID:
      - count, mean, std, median, p95, p99
    """
    group = (
        df_errors
        .groupby("ShipTypeID")["reconstruction_error"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            p95=lambda x: np.percentile(x, 95),
            p99=lambda x: np.percentile(x, 99),
        )
        .reset_index()
    )
    return group


def compute_featurewise_mse(
    model: torch.nn.Module,
    df_seq: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate the mean MSE per feature, aggregated over the entire df_seq.
    """
    dataset = AISTrajectoryDataset(df_seq, return_meta=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    sum_sq_error_per_feat: Optional[torch.Tensor] = None
    count_elements = 0

    with torch.no_grad():
        for x, ship_type_id, _, _ in loader:
            x = x.to(device)                       # (B, T, F)
            ship_type_id = ship_type_id.to(device) # (B,)

            recon, _ = model(x, ship_type_id)      # (B, T, F)
            mse = F.mse_loss(recon, x, reduction="none")  # (B, T, F)
            mse_sum = mse.sum(dim=(0, 1))                 # (F,)

            if sum_sq_error_per_feat is None:
                sum_sq_error_per_feat = mse_sum.cpu()
            else:
                sum_sq_error_per_feat += mse_sum.cpu()

            count_elements += x.shape[0] * x.shape[1]

    mean_mse_per_feat = (sum_sq_error_per_feat / count_elements).numpy()

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(len(mean_mse_per_feat))]

    df_feat = pd.DataFrame(
        {
            "feature": feature_names,
            "mse": mean_mse_per_feat,
        }
    )
    return df_feat



def choose_threshold_from_train(
    df_train_errors: pd.DataFrame,
    percentile: float = 99.5,
) -> float:
    """
    Choose a threshold based on training errors (e.g., 99.5th percentile).
    """
    errs = df_train_errors["reconstruction_error"].values
    thr = float(np.percentile(errs, percentile))
    return thr


def apply_threshold(
    df_errors: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Add a column 'is_anomaly' based on the threshold.
    """
    df = df_errors.copy()
    df["is_anomaly"] = df["reconstruction_error"] > threshold
    return df


def compute_flag_rate(df_errors: pd.DataFrame) -> float:
    """
    Percentage of segments flagged as anomalies.
    """
    if "is_anomaly" not in df_errors.columns:
        raise ValueError("The column 'is_anomaly' is not present in the DataFrame.")
    return float(df_errors["is_anomaly"].mean())


# (Optional) metrics with true labels (not needed for now, you can ignore)
# try:
#     

#     def evaluate_with_labels(
#         df_errors: pd.DataFrame,
#         label_col: str = "is_anomaly_gt",
#     ) -> Dict[str, float]:
#         """
#         If you have a column 'is_anomaly_gt' (0/1), evaluate:
#           - ROC AUC
#           - PR AUC (average precision)
#         """
#         if label_col not in df_errors.columns:
#             raise ValueError(f"Column '{label_col}' not found in the DataFrame.")

#         y_true = df_errors[label_col].astype(int).values
#         scores = df_errors["reconstruction_error"].values

#         roc_auc = roc_auc_score(y_true, scores)
#         pr_auc = average_precision_score(y_true, scores)

#         return {
#             "roc_auc": float(roc_auc),
#             "pr_auc": float(pr_auc),
#         }

# except ImportError:
#     def evaluate_with_labels(*args, **kwargs):
#         raise ImportError("To use evaluate_with_labels, scikit-learn is required.")


# ======================================================
# PART B – PHASE 1: Training multi-model on train/val
# ======================================================

@dataclass
class AEConfig:
    hidden_dim: int = 128
    latent_dim: int = 64
    shiptype_emb_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 10
    max_grad_norm: float = 1.0
    threshold_percentile: float = 99.5
    run_name: str = ""  # to identify the run (file name, etc.)

    # --- Early stopping ---
    use_early_stopping: bool = True
    early_stopping_patience: int = 5       # how many epochs without improvement
    early_stopping_min_delta: float = 1e-4 # minimum improvement on val_loss


def build_model_from_config(
    config: AEConfig,
    n_features: int,
    num_shiptypes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create an TrainModel from the config.
    (It is assumed that the class is defined elsewhere.)
    """
    model = TrainModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_shiptypes=num_shiptypes,
        shiptype_emb_dim=config.shiptype_emb_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    return model


def train_autoencoder_with_config(
    df_seq_train: pd.DataFrame,
    df_seq_val: pd.DataFrame,
    config: AEConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Training loop su train/val con AEConfig + early stopping su val_loss.
    """
    # Dataset / Dataloader
    train_dataset = AISTrajectoryDataset(df_seq_train)  # return_meta=False
    val_dataset   = AISTrajectoryDataset(df_seq_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Inferisci dimensioni
    sample_x, _ = train_dataset[0]
    _, n_features = sample_x.shape
    num_shiptypes = df_seq_train["ShipTypeID"].nunique()

    model = build_model_from_config(
        config=config,
        n_features=n_features,
        num_shiptypes=num_shiptypes,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # --- Early stopping state ---
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0

        for x, ship_type_id in train_loader:
            x = x.to(device)
            ship_type_id = ship_type_id.to(device)

            recon, _ = model(x, ship_type_id)
            seq_errors = sequence_loss_fn(recon, x)   # (B,)
            loss = seq_errors.mean()

            optimizer.zero_grad()
            loss.backward()
            if config.max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, ship_type_id in val_loader:
                x = x.to(device)
                ship_type_id = ship_type_id.to(device)

                recon, _ = model(x, ship_type_id)
                seq_errors = sequence_loss_fn(recon, x)
                loss = seq_errors.mean()

                val_loss_sum += loss.item() * x.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        print(
            f"[{config.run_name or 'run'}] "
            f"Epoch {epoch}/{config.epochs} - "
            f"train MSE: {train_loss:.6f} - val MSE: {val_loss:.6f}"
        )

        # ---- Aggiorna stato early stopping ----
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        # ---- Check early stopping ----
        if config.use_early_stopping and epochs_no_improve >= config.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best val_loss {best_val_loss:.6f} at epoch {best_epoch})"
            )
            break

    # Se abbiamo trovato un modello migliore durante il training, ricarichiamo quei pesi
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Info aggiuntive nella history (comode da salvare insieme)
    history["best_val_loss"] = [best_val_loss]
    history["best_epoch"] = [best_epoch]

    return model, history



def train_and_eval_on_train_val(
    df_seq_train: pd.DataFrame,
    df_seq_val: pd.DataFrame,
    config: AEConfig,
    device: torch.device,
    save_dir: str = "models",
) -> Dict[str, Any]:
    """
    Phase 1 for ONE config:
      - train a model on train/val
      - compute errors and statistics on train/val
      - choose threshold from train
      - save EVERYTHING:
          - weights (.pt)
          - meta (.json) with config, dim, stats, threshold
          - history losses (.json)
          - df_train_errors (.parquet)
          - df_val_errors (.parquet)

    Returns:
      - summary (riga riassuntiva)
      - paths ai file salvati
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Parametric training ----
    model, history = train_autoencoder_with_config(
        df_seq_train=df_seq_train,
        df_seq_val=df_seq_val,
        config=config,
        device=device,
    )

    # ---- Errors on train/val ----
    df_train_err = evaluate_model_on_df(
        model, df_seq_train, device, batch_size=config.batch_size, set_name="train"
    )
    df_val_err = evaluate_model_on_df(
        model, df_seq_val, device, batch_size=config.batch_size, set_name="val"
    )

    stats_train = summarize_reconstruction_errors(df_train_err, "train")
    stats_val   = summarize_reconstruction_errors(df_val_err, "val")

    threshold = choose_threshold_from_train(
        df_train_errors=df_train_err,
        percentile=config.threshold_percentile,
    )

    df_train_thr = apply_threshold(df_train_err, threshold)
    df_val_thr   = apply_threshold(df_val_err, threshold)

    flagged_train = compute_flag_rate(df_train_thr)
    flagged_val   = compute_flag_rate(df_val_thr)

    # ---- Dimensional info to reconstruct the model ----
    sample_seq = np.array(df_seq_train.iloc[0]["Sequence"], dtype=np.float32)
    n_features = sample_seq.shape[1]
    num_shiptypes = df_seq_train["ShipTypeID"].nunique()

    # ---- Paths for saving ----
    run_name = config.run_name or "run"
    model_path      = os.path.join(save_dir, f"{run_name}.pt")
    meta_path       = os.path.join(save_dir, f"{run_name}_meta.json")
    history_path    = os.path.join(save_dir, f"{run_name}_history.json")
    train_err_path  = os.path.join(save_dir, f"{run_name}_train_errors.parquet")
    val_err_path    = os.path.join(save_dir, f"{run_name}_val_errors.parquet")

    # Model weights
    torch.save(model.state_dict(), model_path)

    # Meta (config + useful info)
    meta = {
        "config": asdict(config),
        "input_dim": n_features,
        "num_shiptypes": int(num_shiptypes),
        "train_stats": stats_train,
        "val_stats": stats_val,
        "threshold": threshold,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # History (list of train_loss / val_loss per epoch)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Errors per segment in parquet (convenient for reloading)
    df_train_err.to_parquet(train_err_path, index=False)
    df_val_err.to_parquet(val_err_path, index=False)

    summary = {
        "run_name": run_name,
        "model_path": model_path,
        "meta_path": meta_path,
        "history_path": history_path,
        "train_err_path": train_err_path,
        "val_err_path": val_err_path,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "shiptype_emb_dim": config.shiptype_emb_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "threshold_percentile": config.threshold_percentile,
        "train_loss_final": history["train_loss"][-1],
        "val_loss_final": history["val_loss"][-1],
        "train_mean_error": stats_train["mean"],
        "val_mean_error": stats_val["mean"],
        "train_p99": stats_train["p99"],
        "val_p99": stats_val["p99"],
        "threshold": threshold,
        "train_flagged_pct": flagged_train,
        "val_flagged_pct": flagged_val,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": history["best_epoch"][0],
        "best_val_loss": history["best_val_loss"][0],
    }

    return {
        "summary": summary,
        "model_path": model_path,
        "meta_path": meta_path,
        "history_path": history_path,
        "train_err_path": train_err_path,
        "val_err_path": val_err_path,
        "df_train_errors": df_train_err,
        "df_val_errors": df_val_err,
    }



def run_training_grid(
    df_seq_train: pd.DataFrame,
    df_seq_val: pd.DataFrame,
    configs: List[AEConfig],
    device: Optional[torch.device] = None,
    save_dir: str = "models",
) -> pd.DataFrame:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries: List[Dict[str, Any]] = []

    for cfg in configs:
        print("\n======================================")
        print(f"Training config: {cfg.run_name or cfg}")
        print("======================================")

        result = train_and_eval_on_train_val(
            df_seq_train=df_seq_train,
            df_seq_val=df_seq_val,
            config=cfg,
            device=device,
            save_dir=save_dir,
        )

        summaries.append(result["summary"])

    df_summaries = pd.DataFrame(summaries)

    # Salvo/aggiorno il file riassuntivo globale
    summary_csv_path = os.path.join(save_dir, "training_summaries.csv")
    df_summaries.to_csv(summary_csv_path, index=False)

    return df_summaries



# ======================================================
# PART C – PHASE 2: Load best model + evaluation on test
# ======================================================

def load_model_from_files(
    model_path: str,
    meta_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, AEConfig, Dict[str, Any]]:
    """
    Load:
      - state_dict of the model
      - meta (config, input_dim, num_shiptypes, threshold, stats)
    and correctly reinstantiate the model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    cfg_dict = meta["config"]
    config = AEConfig(**cfg_dict)

    input_dim = meta["input_dim"]
    num_shiptypes = meta["num_shiptypes"]

    model = TrainModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_shiptypes=num_shiptypes,
        shiptype_emb_dim=config.shiptype_emb_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, meta


def load_model_by_run_name(
    run_name: str,
    save_dir: str = "models",
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, AEConfig, Dict[str, Any]]:
    """
    Useful wrapper that derives model_path and meta_path from run_name.
    It relies on load_model_from_files.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(save_dir, f"{run_name}.pt")
    meta_path  = os.path.join(save_dir, f"{run_name}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    model, config, meta = load_model_from_files(
        model_path=model_path,
        meta_path=meta_path,
        device=device,
    )
    return model, config, meta

def evaluate_on_test_df(
    model: torch.nn.Module,
    df_seq_test: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    threshold: Optional[float] = None,
    feature_names: Optional[List[str]] = None,
    set_name: str = "test",
) -> Dict[str, Any]:
    """
    PHASE 2:
      - compute errors on df_seq_test
      - global statistics 
      - stats per ship type
      - MSE per feature
      - if threshold is not None:
          % of flagged anomalies
    """
    df_test_err = evaluate_model_on_df(
        model, df_seq_test, device, batch_size=batch_size, set_name=set_name
    )

    stats_test = summarize_reconstruction_errors(df_test_err, set_name)
    by_shiptype = summarize_errors_by_shiptype(df_test_err)
    df_feat_mse = compute_featurewise_mse(
        model, df_seq_test, device, batch_size=batch_size, feature_names=feature_names
    )

    df_test_thr = None
    flagged_pct = None
    if threshold is not None:
        df_test_thr = apply_threshold(df_test_err, threshold)
        flagged_pct = compute_flag_rate(df_test_thr)

    return {
        "df_test_errors": df_test_err,
        "df_test_errors_thr": df_test_thr,
        "stats_test": stats_test,
        "errors_by_shiptype": by_shiptype,
        "feature_mse": df_feat_mse,
        "flagged_pct": flagged_pct,
        "threshold_used": threshold,
    }


def evaluate_saved_model_on_test_path(
    run_name: str,
    test_df_path: str,
    save_dir: str = "models",
    device: Optional[torch.device] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    FASE 2 modulare:
      - carica il modello 'run_name' dai file salvati in save_dir
      - carica df_seq_test da test_df_path
      - usa evaluate_on_test_df per calcolare tutte le statistiche

    Ritorna lo stesso dict di evaluate_on_test_df.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Carico modello + meta
    model, config, meta = load_model_by_run_name(
        run_name=run_name,
        save_dir=save_dir,
        device=device,
    )
    threshold = meta.get("threshold", None)

    # 2) Carico df di test (stessa funzione del train, se va bene)
    df_seq_test = training_utils.load_df_seq(test_df_path)

    # 3) Eval
    eval_test = evaluate_on_test_df(
        model=model,
        df_seq_test=df_seq_test,
        device=device,
        batch_size=config.batch_size,
        threshold=threshold,
        feature_names=feature_names,
        set_name="test",
    )

    return eval_test


# ======================================================
# Example usage (you can adapt / move it elsewhere)
# ======================================================

# -----------------------
# PHASE 1 – TRAINING GRID
# -----------------------

def train_phase(configs: Optional[List[AEConfig]] = None, save_dir: str = "models"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df_seq_train, df_seq_val = load_and_split_train_val(
        train_df_path=config_file.PRE_PROCESSING_DF_TRAIN_PATH,
        val_size=0.2,
        random_state=5,
        split_by_mmsi=True,
    )

    if configs is None:
        print("No configs provided, aborting.")
        return

    df_summaries = run_training_grid(
        df_seq_train=df_seq_train,
        df_seq_val=df_seq_val,
        configs=configs,
        device=device,
        save_dir=save_dir,
    )

    print("\n===== TRAINING SUMMARY =====")
    print(df_summaries.sort_values("val_loss_final").reset_index(drop=True))
    best_row = df_summaries.sort_values("val_loss_final").iloc[0]
    best_model_path = best_row["model_path"]
    best_meta_path = best_row["meta_path"]
    print(f"Best model: {best_model_path} with val_loss {best_row['val_loss_final']:.6f}")
    print(f"Meta file: {best_meta_path}")


# ------------------------------
# PHASE 2 – EVALUATION ON TEST DF
# ------------------------------

def test_phase(run_name: str, folder: str = "test_run"):

    # Load pre-processed training DataFrame from file to  ---> df_seq
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, config, meta = load_model_by_run_name(
        run_name=run_name,
        save_dir=folder,
        device=device,
    )

    threshold = meta.get("threshold", None)

    # Load test DataFrame
    df_seq_test = training_utils.load_df_seq(config_file.PRE_PROCESSING_DF_TEST_PATH)

    # Compute reconstruction errors on the test set (df_test_errors)
    df_test_errors = evaluate_model_on_df(
        model=model,
        df_seq=df_seq_test,
        device=device,
        batch_size=config.batch_size,
        set_name="test",
)
    

    # Path al file di metadata per la denormalizzazione
    # (lo stesso che usi già in inspection_utils.denormalize_predictions)
    metadata_path = config_file.PRE_PROCESSING_METADATA_TEST_PATH  # o la tua path reale

    # Directory base dove vuoi salvare tutto quello che riguarda questo run
    base_output_dir = os.path.join("outputs", run_name + "_test")

    # Indici e nomi delle feature nel tuo Sequence (ordine del preprocessing)
    feature_indices = config_file.FEATURE_INDICES  # e.g., [0, 1, 2, 3, 4]
    feature_names = config_file.FEATURE_NAMES


    results = run_test_inspection_pipeline(
        model=model,
        df_seq_test=df_seq_test,
        df_test_errors=df_test_errors,
        device=device,
        seq_loss_fn=sequence_loss_fn,
        metadata_path=metadata_path,
        base_output_dir=base_output_dir,
        feature_indices=feature_indices,
        feature_names=feature_names,
        num_examples=3,    # 3 best/worst time-series plots
        top_k_maps=5,      # 5 best/worst for dedicated maps
    )

    print("Test inspection outputs saved in:")
    for k, v in results["paths"].items():
        print(f"  {k}: {v}")



def inspection(run_name: str, mmsi: Optional[int] = None, segments: Optional[List[int]] = None):
    

    df_denorm = pd.read_parquet(f"outputs/{run_name}_test/test_predictions_denorm.parquet")

    if mmsi is not None:
        # One MMSI
        create_map_for_mmsi(
            df_denorm=df_denorm,
            mmsi=mmsi,
            out_html=f"outputs/{run_name}_test/map_mmsi_{mmsi}.html",
        )

    if segments is not None:
        # Some specific segments
        create_map_for_segments(
            df_denorm=df_denorm,
            segments=segments,
            out_html=f"outputs/{run_name}_test/map_segments_{'_'.join(map(str, segments))}.html",
        )


if __name__ == "__main__":

    configs = [
        AEConfig(
            hidden_dim=128,
            latent_dim=64,
            shiptype_emb_dim=8,
            num_layers=2,
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=128,            
            epochs=40,                 
            max_grad_norm=1.0,
            threshold_percentile=99.5,
            run_name="h128_l64_L2_do0_lr1e-03_lossfn1",
            use_early_stopping=True,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
        ),
    ]

    # base_archs = [
    #     (64, 16),
    #     (64, 32),   
    #     (128, 32),
    #     (128, 64),
    #     (256, 64),
    # ]

    # for hidden_dim, latent_dim in base_archs:
    #     for num_layers in [1, 2]:
    #         for dropout in [0.0, 0.3]:
    #             for lr in [1e-3, 3e-4]:
    #                 run_name = f"h{hidden_dim}_l{latent_dim}_L{num_layers}_do{int(dropout*10)}_lr{lr:.0e}"
    #                 cfg = AEConfig(
    #                     hidden_dim=hidden_dim,
    #                     latent_dim=latent_dim,
    #                     shiptype_emb_dim=8,
    #                     num_layers=num_layers,
    #                     dropout=dropout,
    #                     learning_rate=lr,
    #                     batch_size=128,            
    #                     epochs=40,                 
    #                     max_grad_norm=1.0,
    #                     threshold_percentile=99.5,
    #                     run_name=run_name,
    #                     use_early_stopping=True,
    #                     early_stopping_patience=5,
    #                     early_stopping_min_delta=1e-4,
    #                 )
    #                 configs.append(cfg)

    # print(f"Number of models in the grid: {len(configs)}")
    # print("Configs:")
    # for cfg in configs:
    #     print(cfg)

    train_phase(configs=configs, save_dir="AE_simple")
    test_phase(run_name="h128_l64_L2_do0_lr1e-03_lossfn1", folder="AE_simple")
    inspection(run_name="h128_l64_L2_do0_lr1e-03_lossfn1")