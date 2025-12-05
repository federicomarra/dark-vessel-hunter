# Main train script for a SINGLE configuration

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


def main_train_single():
    
    # --- 1. CONFIGURATION ---

    # Path to pre-processed training data
    PARQUET_FILE = config_file.PRE_PROCESSING_DF_TRAIN_PATH
    TRAIN_OUTPUT_DIR = config_file.TRAIN_OUTPUT_DIR
    LOSS_TYPE = config_file.LOSS_TYPE

    # ensure output directory exists
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    
    SPLIT_TRAIN_VAL_RATIO = config_file.SPLIT_TRAIN_VAL_RATIO
    EPOCHS = config_file.EPOCHS
    PATIENCE = config_file.PATIENCE
    FEATURES = config_file.FEATURE_COLS
    NUM_SHIP_TYPES = config_file.NUM_SHIP_TYPES
    
    # ---------------------------------------------------------
    # HYPERPARAMETER CONFIGURATION
    # ---------------------------------------------------------
    
    top_params = {
        'hidden_dim': config_file.HIDDEN_DIM,       # Capacity of the LSTM
        'latent_dim': config_file.LATENT_DIM,         # Bottleneck
        'num_layers': config_file.NUM_LAYERS,           # Depth
        'lr': config_file.LEARNING_RATE,          # Learning Rate
        'batch_size': config_file.BATCH_SIZE,        # Batch Size
        'dropout': config_file.DROP_OUT           # Regularization
    }

    run_name_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = (f"H{top_params['hidden_dim']}_L{top_params['latent_dim']}_"
                f"Lay{top_params['num_layers']}_lr{top_params['lr']}_"
                f"BS{top_params['batch_size']}_Drop{top_params['dropout']}_{run_name_suffix}_{LOSS_TYPE}")

    config = {
        "run_name": run_name,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "features": FEATURES,
        "num_ship_types": NUM_SHIP_TYPES,
        "shiptype_emb_dim": 8,
        "loss_type": LOSS_TYPE,
        
        # Dynamic Params (ora fissi)
        "hidden_dim": top_params['hidden_dim'],
        "latent_dim": top_params['latent_dim'],
        "num_layers": top_params['num_layers'],
        "lr": top_params['lr'],
        "batch_size": top_params['batch_size'],
        "dropout": top_params['dropout']
    }
    
    configs = [config]

    print(f"Running a single configuration: {run_name}")

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
        save_path = f"{TRAIN_OUTPUT_DIR}/weights_{config['run_name']}.pth"
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
    results_path = os.path.join(TRAIN_OUTPUT_DIR, "results_summary_single"+ datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print result
    print("\n=== Single Configuration Result ===") 
    print(f"Run: {config['run_name']} | Best Val Loss: {float(best_loss):.6f}")


if __name__ == "__main__":
    main_train_single()
