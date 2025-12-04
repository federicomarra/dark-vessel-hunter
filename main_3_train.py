# Main train script

# File imports
import config as config_file
from src.train.ais_dataset import AISDataset, ais_collate_fn
from src.train.model import AIS_LSTM_Autoencoder
from src.train.training_loop import run_experiment

# Library imports
import datetime
import torch
from torch.utils.data import DataLoader, random_split
import os
import json
import itertools # Added for grid search


def main_train():
    
    # --- 1. CONFIGURATION ---

    # Path to pre-processed training data
    PARQUET_FILE = config_file.PRE_PROCESSING_DF_TRAIN_PATH
    TRAIN_OUTPUT_DIR = config_file.TRAIN_OUTPUT_DIR

    # ensure output directory exists
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    
    SPLIT_TRAIN_VAL_RATIO = config_file.SPLIT_TRAIN_VAL_RATIO
    EPOCHS = config_file.EPOCHS
    PATIENCE = config_file.PATIENCE
    FEATURES = config_file.FEATURE_COLS
    NUM_SHIP_TYPES = config_file.NUM_SHIP_TYPES
    LOSS_TYPE = config_file.LOSS_TYPE

    
    # ---------------------------------------------------------
    # HYPERPARAMETER GRID SEARCH
    # ---------------------------------------------------------
    # Define ranges for grid search
    # Since you have high compute power, we explore Width (hidden) vs Depth (layers)
    # and Bottleneck tightness (latent).
    
    param_grid = {
        'hidden_dim': [128, 256],       # Capacity of the LSTM
        'latent_dim': [16, 64],         # Bottleneck: 16 (Anomaly Detection) vs 64 (Reconstruction)
        'num_layers': [1, 2],           # Depth
        'lr': [0.001, 0.0001],          # Learning Rate
        'batch_size': [64, 128],        # Batch Size
        'dropout': [0.0, 0.2]           # Regularization
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
                    f"BS{params['batch_size']}_Drop{params['dropout']}_{LOSS_TYPE}")
        
        config = {
            "run_name": run_name,
            "epochs": EPOCHS,              # Fixed epochs
            "patience": PATIENCE,             # Fixed patience
            "features": FEATURES,
            "num_ship_types": NUM_SHIP_TYPES,
            "shiptype_emb_dim": 8,     # Keep embedding dim constant for now
            "loss_type": LOSS_TYPE,
            
            # Dynamic Params
            "hidden_dim": params['hidden_dim'],
            "latent_dim": params['latent_dim'],
            "num_layers": params['num_layers'],
            "lr": params['lr'],
            "batch_size": params['batch_size'],
            "dropout": params['dropout']
        }
        configs.append(config)

    print(f"Generated {len(configs)} unique configurations for training using {LOSS_TYPE} loss.")


    # --- 2. DEVICE SETUP ---
    if torch.cuda.is_available():
        device = torch.device("cuda")  # for PC with NVIDIA
        print(f"Using device: {device} (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # for Mac Apple Silicon
        print(f"Using device: {device} (Apple GPU)")
    else:
        device = torch.device("cpu")   # Fallback on CPU
        print(f"Using device: {device} (CPU)")


    # --- 3. DATA LOAD ---
    if not os.path.exists(PARQUET_FILE):
        print(f"Error: {PARQUET_FILE} not found.")
        return

    # Initialize Dataset
    full_dataset = AISDataset(PARQUET_FILE)
    input_dim = full_dataset.input_dim

    # Split Train/Val (80/20)
    train_size = int(SPLIT_TRAIN_VAL_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


    # --- 4. EXPERIMENT LOOP ---
    results = []

    for config in configs:
        # Create DataLoaders
        train_loader = DataLoader(  # Training DataLoader
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=ais_collate_fn
        )
        
        val_loader = DataLoader(    # Validation DataLoader
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
        
        # Save results
        results.append({
            "config": config['run_name'],
            "best_val_loss": best_loss,
            "history": history
        })

        # Save model and config
        os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
        with open(f"{TRAIN_OUTPUT_DIR}/config_{config['run_name']}.json", 'w') as f:
            json.dump(config, f, indent=4)


    # --- 5. SUMMARY OF THE MODEL---
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
    main_train()
