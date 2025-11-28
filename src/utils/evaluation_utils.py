import os
from typing import Tuple, Dict, Any
from xml.parsers.expat import model
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import config

PLOT_PATH = config.PLOT_PATH
PREDICTION_DF_PATH = config.PREDICTION_DF_PATH


class AISTrajectoryDataset(Dataset):
    """
    Dataset wrapping df_seq:
      - Sequence: list-of-lists (T, F) or np.ndarray
      - ShipTypeID: integer class id
    """

    def __init__(self, df_seq: pd.DataFrame):
        self.df = df_seq.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = np.array(row["Sequence"], dtype=np.float32)  # (T, F)
        x = torch.from_numpy(seq)                         # (T, F)

        ship_type_id = int(row["ShipTypeID"])
        ship_type_id = torch.tensor(ship_type_id, dtype=torch.long)

        return x, ship_type_id
    

### ------- PLOTS --------- ###

def make_plots(
    model: nn.Module,
    test_loader: DataLoader,
    #df_seq: pd.DataFrame,
    device: torch.device,
    seq_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Dict[str, Any]:
    print("\n=== Reconstruction errors on test ===")

    model.eval()
    test_scores = []

    with torch.no_grad():
        for x, ship_type_id in test_loader:
            x = x.to(device)
            ship_type_id = ship_type_id.to(device)

            recon, _ = model(x, ship_type_id)   # (B, T, F)
            seq_error = seq_loss_fn(recon, x)    # (B,)
            test_scores.append(seq_error.cpu().numpy())

    test_scores = np.concatenate(test_scores, axis=0)



    summarize_scores("Test", test_scores)
    # ---- Plot reconstruction error histograms ----
    plot_histogram(
        test_scores,
        title="Test reconstruction error",
        filename=f"{PLOT_PATH}/test_recon_hist.png",
    )

    # SYNTETHETIC ANOMALIES (TRUE NEGATIVE / POSITIVE EVALUATION)

    # print("\n=== Synthetic anomaly evaluation ===")
    # X_normal, X_anom, ship_ids = build_synth_data_with_shiptype(df_seq, max_samples=2000)
    # metrics = evaluate_on_synthetic_anomalies(
    #     model,
    #     X_normal, X_anom,
    #     ship_ids,
    #     device,
    # )
    # print(f"ROC-AUC (normal vs synthetic): {metrics['roc_auc']:.4f}")
    # print(f"PR-AUC  (normal vs synthetic): {metrics['pr_auc']:.4f}")

    # # Plot histogram normal vs synthetic scores
    # plot_histogram_two(
    #     metrics["scores_normal"],
    #     metrics["scores_anom"],
    #     labels=("normal", "synthetic"),
    #     title="Reconstruction error: normal vs synthetic",
    #     filename=f"{PLOT_PATH}/synthetic_scores_hist.png",
    # )

    # # Plot ROC curve
    # plot_roc_curve_fig(
    #     metrics["fpr"],
    #     metrics["tpr"],
    #     metrics["roc_auc"],
    #     filename=f"{PLOT_PATH}/roc_curve_synthetic.png",
    # )

    # # Plot PR curve
    # plot_pr_curve_fig(
    #     metrics["recall"],
    #     metrics["precision"],
    #     metrics["pr_auc"],
    #     filename=f"{PLOT_PATH}/pr_curve_synthetic.png",
    # )

    # ---- Plot ispezione feature per segmenti reali e sintetici ----
    # assumiamo prime 4 feature: lat, lon, sog, cog
    feature_indices = [0, 1, 2, 3]
    feature_names = {
        0: "Latitude",
        1: "Longitude",
        2: "SOG",
        3: "COG",
    }

    # reali: prendiamo dataset di test
    test_dataset = test_loader.dataset  # AISTrajectoryDataset
    plot_inspection_real_sequences(
        model,
        test_dataset,
        test_scores,
        device,
        feature_indices,
        feature_names,
        num_examples=3,
    )

    # # sintetici: prendiamo top anomalie
    # plot_inspection_synthetic_examples(
    #     model,
    #     X_normal,
    #     X_anom,
    #     ship_ids,
    #     metrics,
    #     device,
    #     feature_indices,
    #     feature_names,
    #     num_examples=3,
    # )

    # Suggested threshold (can be tuned): 99th percentile of validation scores
    threshold_99 = np.quantile(test_scores, 0.99)
    print(f"\nSuggested anomaly threshold (99th percentile of test): {threshold_99:.6f}")
    print(f'Plots saved in directory: "{PLOT_PATH}"')

    results = {
        "test_scores": test_scores,
        #"synthetic_metrics": metrics,
        "threshold_99": threshold_99,
    }
    return results


### ------- PREDICTIONS DATAFRAME --------- ###

def build_predictions_df(
    model: nn.Module,
    dataset: AISTrajectoryDataset,
    device: torch.device,
    seq_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> pd.DataFrame:
    """
    Builds a DataFrame with:
      - Segment_nr
      - MMSI
      - ShipTypeID
      - Sequence_real : list of lists (T, F)
      - Sequence_pred : list of lists (T, F)
      - recon_error   : mean MSE over T and F

    for all segments in the dataset (typically validation).
    """
    model.eval()
    records = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            x, ship_type_id = dataset[idx]          # x: (T, F)
            x_batch = x.unsqueeze(0).to(device)     # (1, T, F)
            st_batch = ship_type_id.unsqueeze(0).to(device)

            recon_batch, _ = model(x_batch, st_batch)

            seq_error_batch = seq_loss_fn(recon_batch, x_batch)  # (1,)
            seq_error = seq_error_batch.mean().item()

            recon = recon_batch.squeeze(0).cpu().numpy()   # (T, F)
            x_np = x.numpy()                               # (T, F)

            # prendi metadata originali dal df interno del dataset
            row_meta = dataset.df.iloc[idx]
            seg_nr = int(row_meta["Segment_nr"])
            mmsi = row_meta["MMSI"]
            ship_type = int(row_meta["ShipTypeID"])

            records.append({
                "Segment_nr": seg_nr,
                "MMSI": mmsi,
                "ShipTypeID": ship_type,
                "Sequence_real": x_np.tolist(),
                "Sequence_pred": recon.tolist(),
                "recon_error": seq_error,
            })

    df_pred = pd.DataFrame(records)
    return df_pred






























### ------ HELPER FUNCTIONS FOR EVALUATION ------ ###


    # ---------------------------------------------------------------------------
    # Plot helpers
    # ---------------------------------------------------------------------------

def plot_sequence_real_vs_recon(
    x_seq: np.ndarray,
    recon_seq: np.ndarray,
    feature_indices,
    feature_names: Dict[int, str],
    title: str,
    filename: str,
) -> None:
    """
    Inspection plot for a real segment:
    - x_seq: (T, F) real input
    - recon_seq: (T, F) reconstruction
    - feature_indices: list of feature indices to plot (e.g. [0,1,2,3])
    - feature_names: dict {idx: name}
    """
    ensure_dir(os.path.dirname(filename) or ".")
    T = x_seq.shape[0]
    t = np.arange(T)
    n_feats = len(feature_indices)

    plt.figure(figsize=(8, 2.5 * n_feats))
    for i, fidx in enumerate(feature_indices):
        plt.subplot(n_feats, 1, i + 1)
        plt.plot(t, x_seq[:, fidx], label="real")
        plt.plot(t, recon_seq[:, fidx], label="recon", linestyle="--")
        plt.ylabel(feature_names.get(fidx, f"feat_{fidx}"))
        if i == 0:
            plt.title(title)
        if i == n_feats - 1:
            plt.xlabel("timestep")
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_sequence_synthetic_three(
    x_normal: np.ndarray,
    x_anom: np.ndarray,
    recon_anom: np.ndarray,
    feature_indices,
    feature_names: Dict[int, str],
    title: str,
    filename: str,
) -> None:
    """
    Inspection plot for synthetic anomalies:
    - x_normal: original segment (normal)
    - x_anom: corrupted segment (model input)
    - recon_anom: reconstruction of the corrupted segment
    """
    ensure_dir(os.path.dirname(filename) or ".")
    T = x_normal.shape[0]
    t = np.arange(T)
    n_feats = len(feature_indices)

    plt.figure(figsize=(8, 2.5 * n_feats))
    for i, fidx in enumerate(feature_indices):
        plt.subplot(n_feats, 1, i + 1)
        plt.plot(t, x_normal[:, fidx], label="normal (orig)", alpha=0.7)
        plt.plot(t, x_anom[:, fidx], label="input (corrupted)", alpha=0.7)
        plt.plot(t, recon_anom[:, fidx], label="recon(corrupted)", linestyle="--")
        plt.ylabel(feature_names.get(fidx, f"feat_{fidx}"))
        if i == 0:
            plt.title(title)
        if i == n_feats - 1:
            plt.xlabel("timestep")
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_inspection_real_sequences(
    model: nn.Module,
    val_dataset: AISTrajectoryDataset,
    val_scores: np.ndarray,
    device: torch.device,
    feature_indices,
    feature_names: Dict[int, str],
    num_examples: int = 3,
) -> None:
    """
    Plots some real segments from the validation set:
    - the best ones (low error)
    - the worst ones (high error)
    Each figure contains the selected features, input vs recon.
    The segment number is included in the title and filename.
    """
    ensure_dir(f"{PLOT_PATH}/real_sequences")

    # sort by ascending error
    idx_sorted = np.argsort(val_scores)
    low_idxs = idx_sorted[:num_examples]
    high_idxs = idx_sorted[-num_examples:]

    def plot_one(idx: int, kind: str):
        x, ship_type_id = val_dataset[idx]
        x_batch = x.unsqueeze(0).to(device)
        ship_type_id_batch = ship_type_id.unsqueeze(0).to(device)

        with torch.no_grad():
            recon_batch, _ = model(x_batch, ship_type_id_batch)

        x_np = x.numpy()
        recon_np = recon_batch.squeeze(0).cpu().numpy()

        seg_nr = int(val_dataset.df.iloc[idx]["Segment_nr"])
        title = f"Segment {seg_nr} ({kind} error)"
        filename = f"{PLOT_PATH}/real_sequences/segment_{seg_nr}_{kind}.png"

        plot_sequence_real_vs_recon(
            x_np,
            recon_np,
            feature_indices,
            feature_names,
            title,
            filename,
        )

    for idx in low_idxs:
        plot_one(int(idx), "low")
    for idx in high_idxs:
        plot_one(int(idx), "high")


def plot_inspection_synthetic_examples(
    model: nn.Module,
    X_normal: np.ndarray,
    X_anom: np.ndarray,
    ship_ids: np.ndarray,
    metrics: Dict[str, Any],
    device: torch.device,
    feature_indices,
    feature_names: Dict[int, str],
    num_examples: int = 3,
) -> None:
    """
    Plots some synthetic examples:
    - takes the N with highest error in synthetic
    - shows normal vs corrupted vs recon(corrupted)
    """
    ensure_dir(f"{PLOT_PATH}/synthetic_sequences")

    scores_anom = metrics["scores_anom"]
    idx_sorted = np.argsort(scores_anom)
    high_idxs = idx_sorted[-num_examples:]

    for rank, idx in enumerate(high_idxs):
        idx = int(idx)
        x_norm = X_normal[idx]   # (T, F)
        x_anom = X_anom[idx]     # (T, F)

        x_anom_t = torch.tensor(x_anom, dtype=torch.float32, device=device).unsqueeze(0)
        ship_id_t = torch.tensor([ship_ids[idx]], dtype=torch.long, device=device)

        with torch.no_grad():
            recon_anom_t, _ = model(x_anom_t, ship_id_t)
        recon_anom = recon_anom_t.squeeze(0).cpu().numpy()

        title = f"Synth anomaly idx={idx} (rank {rank+1})"
        filename = f"{PLOT_PATH}/synthetic_sequences/synth_{idx}_top{rank+1}.png"

        plot_sequence_synthetic_three(
            x_norm,
            x_anom,
            recon_anom,
            feature_indices,
            feature_names,
            title,
            filename,
        )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_histogram(scores: np.ndarray, title: str, filename: str) -> None:
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure()
    plt.hist(scores, bins=50)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_histogram_two(
    scores1: np.ndarray,
    scores2: np.ndarray,
    labels: Tuple[str, str],
    title: str,
    filename: str,
    bins: int = 50,
) -> None:
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure()
    plt.hist(scores1, bins=bins, alpha=0.5, density=True, label=labels[0])
    plt.hist(scores2, bins=bins, alpha=0.5, density=True, label=labels[1])
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_roc_curve_fig(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    filename: str,
) -> None:
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (normal vs synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pr_curve_fig(
    recall: np.ndarray,
    precision: np.ndarray,
    pr_auc: float,
    filename: str,
) -> None:
    ensure_dir(os.path.dirname(filename) or ".")
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve (normal vs synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def summarize_scores(name: str, scores: np.ndarray) -> None:
    print(f"{name}:")
    print(f"  mean: {scores.mean():.6f}")
    print(f"  std:  {scores.std():.6f}")
    print(f"  q95:  {np.quantile(scores, 0.95):.6f}")
    print(f"  q99:  {np.quantile(scores, 0.99):.6f}")
    print(f"  max:  {scores.max():.6f}")