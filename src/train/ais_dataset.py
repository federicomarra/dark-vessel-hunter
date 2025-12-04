import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class AISDataset(Dataset):
    def __init__(self, parquet_path, features=None):
        """
        Args:
            parquet_path (str): Path to the parquet file.
            features (list): List of feature column names to use.
        """
        if features is None:
            features = ['Latitude', 'Longitude', 'SOG', 'COG_sin', 'COG_cos']
            
        self.features = features
        
        print(f"Loading data from {parquet_path}...")
        # Load data
        df = pd.read_parquet(parquet_path)
        
        # Ensure data is sorted by segment and timestamp
        df = df.sort_values(by=['Segment_nr', 'Timestamp'])
        
        # Normalize Continuous Features
        # We fit a scaler on the whole dataset. 
        # Note: In a real prod scenario, fit scaler on TRAIN split only and transform TEST.
        print("Normalizing features...")
        self.scaler = MinMaxScaler()
        df[self.features] = self.scaler.fit_transform(df[self.features].astype(np.float32))
                
        # Group by Segment_nr
        print("Grouping segments...")
        self.grouped_data = []
        
        # Iterate over groups (Segment_nr)
        # We convert to numpy first for speed
        for segment_id, group in df.groupby('Segment_nr'):
            seq_data = group[self.features].values
            shiptype = group['ShipTypeID'].iloc[0] # Assumes shiptype is constant per segment
            
            # Convert to tensors
            self.grouped_data.append({
                'sequence': torch.tensor(seq_data, dtype=torch.float32),
                'shiptype': torch.tensor(shiptype, dtype=torch.long),
                'segment_id': segment_id
            })
            
        print(f"Processed {len(self.grouped_data)} unique segments.")

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        return self.grouped_data[idx]
    
    @property
    def input_dim(self):
        return len(self.features)
    
    @property
    def num_ship_types(self):
        # If IDs are 0-4, max is 4, so we need size 5 (indices 0,1,2,3,4)
        return self.max_shiptype_id + 1

def ais_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Returns:
        padded_seqs: (Batch, Max_Len, Features)
        lengths: (Batch) - Actual lengths before padding
        ship_types: (Batch)
    """
    sequences = [item['sequence'] for item in batch]
    ship_types = torch.stack([item['shiptype'] for item in batch])
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences with 0.0 (batch_first=True)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    return padded_seqs, lengths, ship_types