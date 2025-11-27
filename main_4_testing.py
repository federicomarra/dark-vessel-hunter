# Main testing script

# File imports
import config
from main_3_training import LSTMVAEWithShipType  
from src.pre_proc.pre_processing_utils import add_delta_t, split_segments_fixed_length, one_hot_encode_nav_status, label_ship_types
from src.pre_proc.ais_query import query_ais_duckdb

# Library imports
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

TEST_OUTPUT_CSV = config.TEST_OUTPUT_CSV

def load_training_artifacts():
    df_train = pd.read_parquet(config.PRE_PROCESSING_DF_PATH)

    NUMERIC_COLS = config.NUMERIC_COLS
    NAV_ONEHOT_COLS = [c for c in df_train.columns if c.startswith("NavStatus_")]
    FEATURE_COLS = NUMERIC_COLS + NAV_ONEHOT_COLS

    # Load normalization metadata
    with open(config.PRE_PROCESSING_METADATA_PATH, "r") as f:
        meta = json.load(f)
    mean_arr = meta["mean"]
    std_arr = meta["std"]

    # Model hyperparameters
    T = config.SEGMENT_MAX_LENGTH
    F = len(FEATURE_COLS)

    # Load model weights
    state_dict = torch.load(config.MODEL_PATH, map_location="cpu")

    # Infer number of ship types from embedding shape
    emb_w = state_dict["shiptype_emb.weight"]
    num_shiptypes = emb_w.shape[0]

    model = LSTMVAEWithShipType(
        input_dim=F,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_shiptypes=num_shiptypes,
        shiptype_emb_dim=config.SHIPTYPE_EMB_DIM,
        num_layers=config.NUM_LAYERS,
    )
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully with input_dim =", F)
    return FEATURE_COLS, NUMERIC_COLS, NAV_ONEHOT_COLS, mean_arr, std_arr, model, T

def preprocess_unseen_day(date, FEATURE_COLS, NUMERIC_COLS, NAV_ONEHOT_COLS, mean_arr, std_arr, T):
    """
    Query and preprocess an unseen day.
    Returns:
        df_test_preprocessed
    """
    print(f"Querying raw parquet for date = {date}")
    df = query_ais_duckdb(
        root_path=config.RAW_PARQUET_ROOT,
        dates=[date],
        columns=None,
        verbose=False
    )
    if df.empty:
        raise RuntimeError(f"No AIS rows found for date: {date}")

    # Sorting for DeltaT and splitting
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)
    df = add_delta_t(df)
    df = split_segments_fixed_length(df, max_len=T)

    # One-hot encoding of nav status + align to training
    df, _ = one_hot_encode_nav_status(df)
    df, _ = label_ship_types(df)

    # Ensure all training one-hot columns exist
    for c in NAV_ONEHOT_COLS:
        if c not in df.columns:
            df[c] = 0
    # Remove unseen categories
    for c in [x for x in df.columns if x.startswith("NavStatus_")]:
        if c not in NAV_ONEHOT_COLS:
            df.drop(columns=[c], inplace=True)

    # Normalize using training stats
    for i, col in enumerate(NUMERIC_COLS):
        if col in df.columns:
            mean = mean_arr[i]
            std = std_arr[i] if abs(std_arr[i]) > 1e-12 else 1.0
            df[col] = (df[col] - mean) / std
    print("Preprocessing completed!")
    return df

class AISTestDataset(Dataset):
    def __init__(self, df, feature_cols, T):
        rows = []
        for seg_id, g in df.groupby("Segment_nr"):
            g = g.sort_values("Timestamp")
            if len(g) != T:
                continue
            seq = g[feature_cols].to_numpy(dtype=np.float32)
            rows.append({
                "Sequence": seq,
                "ShipTypeID": int(g["ShipTypeID"].iloc[0]),
                "MMSI": int(g["MMSI"].iloc[0]),
                "StartTS": str(g["Timestamp"].iloc[0]),
                "EndTS": str(g["Timestamp"].iloc[-1]),
                "Segment_nr": int(seg_id),
            })

        self.df_seq = pd.DataFrame(rows).reset_index(drop=True)
        self.T = T
        self.F = len(feature_cols)

    def __len__(self):
        return len(self.df_seq)

    def __getitem__(self, idx):
        row = self.df_seq.iloc[idx]
        x = torch.tensor(row["Sequence"], dtype=torch.float32)
        st = torch.tensor(row["ShipTypeID"], dtype=torch.long)
        ctx = {
            "MMSI": int(row["MMSI"]),
            "StartTS": str(row["StartTS"]) if row.get("StartTS") is not None else None,
            "EndTS": str(row["EndTS"]) if row.get("EndTS") is not None else None,
            "Segment_nr": int(row["Segment_nr"]),
        }
        return x, st, ctx

def run_inference(df_test, FEATURE_COLS, model, T):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = AISTestDataset(df_test, FEATURE_COLS, T)
    loader = DataLoader(dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    all_mse = []
    all_ctx = []
    all_mse_feat = []

    with torch.no_grad():
        for x, st, ctx in loader:
            x = x.to(device)
            st = st.to(device)
            recon, _, _ = model(x, st)

            mse_b = ((x - recon) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()  # (B,)
            all_mse.append(mse_b)
            mse_feat_b = ((x - recon) ** 2).mean(dim=1).detach().cpu().numpy()  # (B, F)
            all_mse_feat.append(mse_feat_b)
            if isinstance(ctx, dict):
                bsz = len(next(iter(ctx.values())))
                for i in range(bsz):
                    all_ctx.append({k: ctx[k][i] for k in ctx})
            else:
                all_ctx.extend(ctx)

    scores = np.concatenate(all_mse)
    mse_feat = np.concatenate(all_mse_feat, axis=0)      # (N, F)

    out = pd.DataFrame(all_ctx)
    out["MSE"] = scores
    feat_sum = mse_feat.sum(axis=1, keepdims=True) + 1e-12
    feat_share = mse_feat / feat_sum 

    top1_idx = mse_feat.argmax(axis=1)                   # (N,)
    out["Top1Feature"] = [FEATURE_COLS[i] for i in top1_idx]
    out["Top1Share"] = feat_share[np.arange(feat_share.shape[0]), top1_idx]

    # Top-3 features (names and shares) as compact strings
    top3_idx = np.argsort(-mse_feat, axis=1)[:, :3]      # (N, 3)
    def _fmt_topk(row_idx):
        pairs = []
        for j in range(top3_idx.shape[1]):
            fi = top3_idx[row_idx, j]
            pairs.append(f"{FEATURE_COLS[fi]}={feat_share[row_idx, fi]:.6f}")
        return "; ".join(pairs)
    out["Top3"] = [ _fmt_topk(i) for i in range(mse_feat.shape[0]) ]

    out = out.sort_values("MSE", ascending=False).reset_index(drop=True)
    return out

def main_testing():
    # Load training artifacts
    FEATURE_COLS, NUMERIC_COLS, NAV_ONEHOT_COLS, mean_arr, std_arr, model, T = load_training_artifacts()

    # Preprocess unseen day
    df_test = preprocess_unseen_day(
        date=config.TEST_DATE,
        FEATURE_COLS=FEATURE_COLS,
        NUMERIC_COLS=NUMERIC_COLS,
        NAV_ONEHOT_COLS=NAV_ONEHOT_COLS,
        mean_arr=mean_arr,
        std_arr=std_arr,
        T=T
    )

    # Run inference
    results = run_inference(df_test, FEATURE_COLS, model, T)

    # Print summary
    print(results["MSE"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    TOPK = 10 if len(results) >= 10 else len(results)
    print(f"\nWorst {TOPK} windows:")
    w_mmsi = max(10, results["MMSI"].astype(str).str.len().max())
    w_seg  = max(8, results["Segment_nr"].astype(str).str.len().max())
    w_feat = max(12, results["Top1Feature"].astype(str).str.len().max())

    for i in range(TOPK):
        r = results.iloc[i]
        print(
            f"#{i+1:02d} "
            f"MMSI={str(r['MMSI']).ljust(w_mmsi)}  "
            f"Seg={str(r['Segment_nr']).ljust(w_seg)}  "
            f"{r['StartTS']} → {r['EndTS']}  "
            f"MSE={r['MSE']:.6f}  "
            f"Feat={str(r['Top1Feature']).ljust(w_feat)}  "
            f"({r['Top1Share']:.2f})"
        )

    # Save CSV
    Path(TEST_OUTPUT_CSV).parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(TEST_OUTPUT_CSV, index=False)
    print(f"Saved results to {TEST_OUTPUT_CSV}")


# Entry point
if __name__ == "__main__":
    main_testing()