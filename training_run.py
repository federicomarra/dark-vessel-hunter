import datetime
import torch
from torch.utils.data import DataLoader, random_split
import os
import json
import itertools # Added for grid search

# Import modules from your files
from src.train.ais_dataset import AISDataset, ais_collate_fn
from src.train.model_anchoring import AIS_LSTM_Autoencoder
from src.train.training_loop import run_experiment

import config as config_file

def training_run():
    # --- 1. CONFIGURATION ---

    # Path to pre-processed training data
    PARQUET_FILE = "ais-data/df_preprocessed/pre_processed_df_train_20min_12h_split.parquet"
    TRAIN_OUTPUT_DIR = "models/20min_12h_split_6months"

    # ensure output directory exists
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    
    FEATURES = config_file.FEATURE_COLS
    NUM_SHIP_TYPES = config_file.NUM_SHIP_TYPES
    
    # ---------------------------------------------------------
    # HYPERPARAMETER GRID SEARCH
    # ---------------------------------------------------------
    # Define ranges for grid search
    # Since you have high compute power, we explore Width (hidden) vs Depth (layers)
    # and Bottleneck tightness (latent).
    
    param_grid = {
        'hidden_dim': [256],       # Capacity of the LSTM
        'latent_dim': [16,32],         # Bottleneck: 16 (Anomaly Detection) vs 64 (Reconstruction)
        'num_layers': [1,2],           # Depth
        'lr': [0.001],          # Learning Rate
        'batch_size': [64],        # Batch Size
        'dropout': [0.0, 0.3]           # Regularization
    }

    configs = []
    
    # Use itertools.product to create all combinations
    keys, values = zip(*param_grid.items())
    for bundle in itertools.product(*values):
        params = dict(zip(keys, bundle))
        
        # Optimization: Dropout is only useful if num_layers > 1
        # Skip dropout=0.2 if num_layers=1 to avoid duplicate equivalent runs
        if params['num_layers'] == 1 and params['dropout'] > 0:
            continue
            
        # Create a descriptive run name
        run_name = (f"H{params['hidden_dim']}_L{params['latent_dim']}_"
                    f"Lay{params['num_layers']}_lr{params['lr']}_"
                    f"BS{params['batch_size']}_Drop{params['dropout']}")
        
        config = {
            "run_name": run_name,
            "epochs": 50,              # Fixed epochs
            "patience": 7,             # Fixed patience
            "features": FEATURES,
            "num_ship_types": NUM_SHIP_TYPES,
            "shiptype_emb_dim": 8,     # Keep embedding dim constant for now
            
            # Dynamic Params
            "hidden_dim": params['hidden_dim'],
            "latent_dim": params['latent_dim'],
            "num_layers": params['num_layers'],
            "lr": params['lr'],
            "batch_size": params['batch_size'],
            "dropout": params['dropout']
        }
        configs.append(config)

    print(f"Generated {len(configs)} unique configurations for training.")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. LOAD DATA ---
    if not os.path.exists(PARQUET_FILE):
        print(f"Error: {PARQUET_FILE} not found.")
        return

    # Initialize Dataset
    full_dataset = AISDataset(PARQUET_FILE)
    input_dim = full_dataset.input_dim

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- 3. EXPERIMENT LOOP ---
    results = []

    for config in configs:
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=ais_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            collate_fn=ais_collate_fn
        )
        
        # Initialize Model with FIXED num_ship_types
        model = AIS_LSTM_Autoencoder(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            num_layers=config['num_layers'],
            num_ship_types=NUM_SHIP_TYPES, # Always use the fixed constant
            shiptype_emb_dim=config['shiptype_emb_dim'],
            dropout=config['dropout']
        ).to(device)
        
        # Run Pipeline
        history, best_loss = run_experiment(config, model, train_loader, val_loader, device, save_path=f"{TRAIN_OUTPUT_DIR}/weights_{config['run_name']}.pth")
        
        results.append({
            "config": config['run_name'],
            "best_val_loss": best_loss,
            "history": history
        })

        # Save model and config
        os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
        with open(f"{TRAIN_OUTPUT_DIR}/config_{config['run_name']}.json", 'w') as f:
            json.dump(config, f, indent=4)

    # --- 4. SUMMARY ---
    # Save full results to JSON (make sure everything is serializable)
    results_path = os.path.join(TRAIN_OUTPUT_DIR, "results_summary_"+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print only the top 3 configurations (lowest validation loss)
    sorted_results = sorted(results, key=lambda r: float(r["best_val_loss"]))
    top_k = sorted_results[:3]
    print("\n=== Top 3 Configurations ===") 
    for i, res in enumerate(top_k, 1):
        print(f"{i}. Run: {res['config']} | Best Val Loss: {float(res['best_val_loss']):.6f}")

if __name__ == "__main__":
    training_run()