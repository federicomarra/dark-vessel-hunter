import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased. Model saved to {self.path}')

def masked_mse_loss(input, target, lengths):
    """
    Calculates MSE only on valid (non-padded) elements.
    input/target shape: (Batch, Max_Len, Features)
    lengths shape: (Batch)
    """
    mask = torch.zeros_like(input, dtype=torch.bool)
    
    # Create mask based on lengths
    for i, length in enumerate(lengths):
        mask[i, :length, :] = 1
        
    # Standard MSE (reduction='none' to keep shape)
    loss = nn.functional.mse_loss(input, target, reduction='none')
    
    # Zero out loss for padded positions
    masked_loss = loss * mask.float()
    
    # Average over valid elements only
    return masked_loss.sum() / mask.sum()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    batch_losses = []
    
    for padded_seqs, lengths, ship_types in loader:
        padded_seqs = padded_seqs.to(device)
        # lengths stays on CPU for pack_padded_sequence usually, 
        # but for mask creation we might need it. 
        # Our masked_mse_loss handles cpu/gpu, but let's be safe.
        
        ship_types = ship_types.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(padded_seqs, lengths, ship_types)
        
        # Calculate loss
        loss = masked_mse_loss(reconstructed, padded_seqs, lengths)
        
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        
    return np.mean(batch_losses)

def validate(model, loader, device):
    model.eval()
    batch_losses = []
    
    with torch.no_grad():
        for padded_seqs, lengths, ship_types in loader:
            padded_seqs = padded_seqs.to(device)
            ship_types = ship_types.to(device)
            
            reconstructed = model(padded_seqs, lengths, ship_types)
            loss = masked_mse_loss(reconstructed, padded_seqs, lengths)
            batch_losses.append(loss.item())
            
    return np.mean(batch_losses)

def run_experiment(config, model, train_loader, val_loader, device, save_path):
    """
    Runs the full training loop for a specific configuration.
    """
    run_name = config.get('run_name', 'experiment')
    print(f"\n--- Starting Run: {run_name} ---")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Checkpoint path
    early_stopping = EarlyStopping(patience=config['patience'], path=save_path)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    return history, early_stopping.best_loss