import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _create_mask(input, lengths):
    """Helper function to create the padding mask."""
    mask = torch.zeros_like(input, dtype=torch.bool, device=input.device)
    for i, length in enumerate(lengths):
        mask[i, :int(length), :] = True
    return mask

## LOSS FUNCTIONS

def masked_mse_loss(input, target, lengths):
    """
    Calculates MSE only on valid (non-padded) elements.
    input/target shape: (Batch, Max_Len, Features)
    lengths shape: (Batch)
    """
    mask = _create_mask(input, lengths)
    loss = F.mse_loss(input, target, reduction='none')
    masked_loss = loss * mask.float()
    return masked_loss.sum() / mask.sum()

def masked_mae_loss(input, target, lengths):
    """
    Calculates MAE only on valid (non-padded) elements.
    input/target shape: (Batch, Max_Len, Features)
    lengths shape: (Batch)
    """
    mask = _create_mask(input, lengths)
    loss = F.l1_loss(input, target, reduction='none')
    masked_loss = loss * mask.float()
    return masked_loss.sum() / mask.sum()


def masked_weighted_mse_loss(input, target, lengths):
    """
    Calculates feature-weighted MSE only on valid (non-padded) elements.
    Weights are internally defined to emphasize dynamic features (SOG, COG).
    
    Order of Features assumed: 
    [Latitude, Longitude, SOG, COG_sin, COG_cos]
    
    Args:
        input (Tensor): Model reconstruction (Batch, Max_Len, Features).
        target (Tensor): Ground truth sequence (Batch, Max_Len, Features).
        lengths (Tensor): Actual sequence lengths (Batch).
        
    Returns:
        Tensor: The mean weighted MSE loss.
    """
    feature_weights = torch.tensor(
        [1.0, 1.0, 2.0, 2.0, 2.0], 
        dtype=torch.float32, 
        device=input.device 
    ) 

    mask = _create_mask(input, lengths)
    loss_unweighted = F.mse_loss(input, target, reduction='none')
    
    weights_expanded = feature_weights.view(1, 1, -1) 
    weighted_loss = loss_unweighted * weights_expanded
    
    masked_weighted_loss = weighted_loss * mask.float()
    return masked_weighted_loss.sum() / mask.sum()


LOSS_FUNCTIONS = {
    'mse': masked_mse_loss,
    'mae': masked_mae_loss,
    'weighted_mse': masked_weighted_mse_loss
}



def train_one_epoch(model, loader, optimizer, device, loss_fn):
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
        loss = loss_fn(reconstructed, padded_seqs, lengths)
        
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        
    return np.mean(batch_losses)

def validate(model, loader, device, loss_fn):
    model.eval()
    batch_losses = []
    
    with torch.no_grad():
        for padded_seqs, lengths, ship_types in loader:
            padded_seqs = padded_seqs.to(device)
            ship_types = ship_types.to(device)
            
            reconstructed = model(padded_seqs, lengths, ship_types)
            loss = loss_fn(reconstructed, padded_seqs, lengths)
            batch_losses.append(loss.item())
            
    return np.mean(batch_losses)

def run_experiment(config, model, train_loader, val_loader, device, save_path):
    """
    Runs the full training loop for a specific configuration.
    """
    run_name = config.get('run_name', 'experiment')
    loss_type = config.get('loss_type', 'mse')
    if loss_type not in LOSS_FUNCTIONS:
        print(f"ATTENTION: Loss type '{loss_type}' not supported. Fallback to 'mse'.")
        loss_type = 'mse'
        
    loss_fn = LOSS_FUNCTIONS[loss_type]
    print(f"\n--- Starting Run: {run_name} (Loss: {loss_type})---")
    
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Checkpoint path
    early_stopping = EarlyStopping(patience=config['patience'], path=save_path)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = validate(model, val_loader, device, loss_fn)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch [{epoch+1:{len(str(config['epochs']))}}/{config['epochs']}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    return history, early_stopping.best_loss
