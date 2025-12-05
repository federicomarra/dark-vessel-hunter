import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import random
import folium
import json

import config as config_file

from src.train.ais_dataset import AISDataset, ais_collate_fn
from src.train.model_anchoring import AIS_LSTM_Autoencoder

# Set style for plots
sns.set_theme(style="whitegrid")

class AISTester:
    def __init__(self, model_config, model_weights_path, output_dir="test_plots", device=None):
        """
        Args:
            model_config (dict): Configuration dictionary used for training (dims, layers, etc.)
            model_weights_path (str): Path to the .pth file
            output_dir (str): Directory where plots will be saved.
        """
        self.config = model_config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Initialize Model
        self.model = AIS_LSTM_Autoencoder(
            input_dim=len(model_config['features']),
            hidden_dim=model_config['hidden_dim'],
            latent_dim=model_config['latent_dim'],
            num_layers=model_config['num_layers'],
            num_ship_types=model_config['num_ship_types'],
            shiptype_emb_dim=model_config['shiptype_emb_dim'],
            dropout=0.0 # No dropout during testing
        ).to(self.device)
        
        # Load Weights
        print(f"Loading weights from {model_weights_path}...")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()
        
    def load_data(self, parquet_path):
        """Loads the test dataset."""
        self.dataset = AISDataset(parquet_path, features=self.config['features'])
        print(f"Test data loaded: {len(self.dataset)} segments.")
        
    def evaluate(self, filter_ids=None):
        """
        Runs prediction on the loaded dataset.
        Args:
            filter_ids (list, optional): List of Segment_nr strings to process during inference. 
                                         If None, processes all.
        """
        loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], collate_fn=ais_collate_fn, shuffle=False)
        
        results = []
        mse_criterion = nn.MSELoss(reduction='none')
        
        print("Running predictions...")
        with torch.no_grad():
            for batch in loader:
                # Unpack batch (handled by collate_fn)
                padded_seqs, lengths, ship_types = batch
                
                # Move to device
                padded_seqs = padded_seqs.to(self.device)
                ship_types = ship_types.to(self.device)
                
                # Predict
                reconstructed = self.model(padded_seqs, lengths, ship_types)
                
                # Calculate errors per element
                # shape: (Batch, Seq, Features)
                raw_errors = mse_criterion(reconstructed, padded_seqs)
                
                # Process batch to extract individual segment results
                batch_size = padded_seqs.size(0)
                
                start_idx = len(results)
                
                for i in range(batch_size):
                    # Global index in dataset
                    global_idx = start_idx + i
                    segment_info = self.dataset[global_idx]
                    seg_id = segment_info['segment_id']
                    
                    # Filtering Logic (Inference level)
                    if filter_ids is not None and seg_id not in filter_ids:
                        continue
                        
                    length = lengths[i].item()
                    
                    # Extract valid data (remove padding)
                    original = padded_seqs[i, :length, :].cpu().numpy()
                    recon = reconstructed[i, :length, :].cpu().numpy()
                    error_per_feat = raw_errors[i, :length, :].mean(dim=0).cpu().numpy() # Mean over time
                    total_mse = raw_errors[i, :length, :].mean().item() # Scalar mean
                    
                    # Inverse Transform to get real units
                    original_real = self.dataset.scaler.inverse_transform(original)
                    recon_real = self.dataset.scaler.inverse_transform(recon)
                    
                    results.append({
                        'segment_id': seg_id,
                        'mse': total_mse,
                        'mse_per_feature': error_per_feat,
                        'original_real': original_real,
                        'recon_real': recon_real,
                        'length': length
                    })
                    
        self.results_df = pd.DataFrame(results)
        print(f"Evaluation complete. Processed {len(self.results_df)} segments.")
        return self.results_df

    def plot_error_distributions(self, filter_ids=None, filename_suffix=""):
        """
        Plots the distribution of reconstruction errors.
        Args:
            filter_ids (list, optional): Filter the existing results by these IDs before plotting.
            filename_suffix (str, optional): Suffix to add to filename (e.g. "_filtered").
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to plot. Run evaluate() first.")
            return

        # Prepare Data
        if filter_ids is not None:
            plot_df = self.results_df[self.results_df['segment_id'].isin(filter_ids)]
            if plot_df.empty:
                print("No segments match the provided filter_ids.")
                return
            title_extra = " (Filtered)"
        else:
            plot_df = self.results_df
            title_extra = ""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Reconstruction Error Distributions (MSE){title_extra}', fontsize=16)
        
        # 1. General MSE Distribution
        sns.histplot(plot_df['mse'], kde=True, ax=axes[0, 0], color='blue')
        axes[0, 0].set_title('Overall Segment MSE')
        axes[0, 0].set_xlabel('Mean Squared Error')
        
        # 2. Feature-wise Distributions
        features = self.config['features']
        
        # Extract feature errors
        if not plot_df.empty:
            feat_errors = np.vstack(plot_df['mse_per_feature'].values)
            
            for i, feature in enumerate(features):
                row = (i + 1) // 3
                col = (i + 1) % 3
                sns.histplot(feat_errors[:, i], kde=True, ax=axes[row, col], color='green')
                axes[row, col].set_title(f'{feature} MSE')
            
        plt.tight_layout()
        
        filename = f"error_distribution{filename_suffix}.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Error distribution plot saved to: {save_path}")

    def plot_best_worst_segments(self, n=3):
        """Plots the N best and N worst reconstructed segments (Line Plots)."""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to plot.")
            return
            
        # Sort by MSE
        sorted_df = self.results_df.sort_values(by='mse')
        
        best_segments = sorted_df.head(n)
        worst_segments = sorted_df.tail(n)
        
        print(f"\n--- Saving Top {n} Best Reconstructions (Line Plots) ---")
        self._plot_segments_lines(best_segments, title_prefix="BEST")
        
        print(f"\n--- Saving Top {n} Worst Reconstructions (Line Plots) ---")
        self._plot_segments_lines(worst_segments, title_prefix="WORST")

    def _plot_segments_lines(self, segment_df, title_prefix):
        features = self.config['features']
        
        for _, row in segment_df.iterrows():
            seg_id = row['segment_id']
            mse = row['mse']
            orig = row['original_real']
            recon = row['recon_real']
            length = row['length']
            time_steps = np.arange(length)
            
            fig, axes = plt.subplots(1, len(features), figsize=(20, 4))
            fig.suptitle(f"{title_prefix}: Segment {seg_id} (MSE: {mse:.5f})", fontsize=14)
            
            for i, feature in enumerate(features):
                ax = axes[i]
                ax.plot(time_steps, orig[:, i], label='Original', color='black', linestyle='-')
                ax.plot(time_steps, recon[:, i], label='Reconstructed', color='red', linestyle='--')
                ax.set_title(feature)
                ax.set_xlabel('Time Step')
                if i == 0:
                    ax.legend()
                    
            plt.tight_layout()
            
            safe_seg_id = str(seg_id).replace("/", "_").replace("\\", "_")
            filename = f"{title_prefix}_seg_{safe_seg_id}.png"
            save_path = os.path.join(self.output_dir, filename)
            
            plt.savefig(save_path)
            plt.close()

    # ==========================================
    # MAPPING FUNCTIONS
    # ==========================================
    def generate_maps(self, n_best_worst=3, n_random=5):
        """Generates HTML maps for Best, Worst, and Random segments."""
        if folium is None:
            print("Skipping map generation (folium not installed).")
            return
        
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to plot. Run evaluate() first.")
            return
            
        sorted_df = self.results_df.sort_values(by='mse')
        
        # 1. Best Segments Map
        best_df = sorted_df.head(n_best_worst)
        self._save_html_map(best_df, f"map_BEST_{n_best_worst}_segments")
        
        # 2. Worst Segments Map
        worst_df = sorted_df.tail(n_best_worst)
        self._save_html_map(worst_df, f"map_WORST_{n_best_worst}_segments")
        
        # 3. Random Segments Map
        if len(sorted_df) > n_random:
            random_df = sorted_df.sample(n=n_random)
        else:
            random_df = sorted_df
        self._save_html_map(random_df, f"map_RANDOM_{n_random}_segments")

    def generate_filtered_map(self, segment_ids, map_name="map_filtered"):
        """Generates an HTML map for specific segment IDs."""
        if folium is None: return
        
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("Run evaluate() first.")
            return
            
        filtered_df = self.results_df[self.results_df['segment_id'].isin(segment_ids)]
        
        if filtered_df.empty:
            print("No matching segments found for map generation.")
            return
            
        self._save_html_map(filtered_df, map_name)

    def _save_html_map(self, segments_df, filename_no_ext):
        """Internal helper to draw map."""
        # Find Lat/Lon indices
        try:
            lat_idx = self.config['features'].index('Latitude')
            lon_idx = self.config['features'].index('Longitude')
        except ValueError:
            print("Error: 'Latitude' or 'Longitude' not in features config. Cannot plot map.")
            return

        # Try to find SOG and COG indices
        try:
            sog_idx = self.config['features'].index('SOG')
        except ValueError:
            sog_idx = -1
            
        try:
            cog_sin_idx = self.config['features'].index('COG_sin')
            cog_cos_idx = self.config['features'].index('COG_cos')
            has_cog = True
        except ValueError:
            has_cog = False

        # Calculate Center
        all_lats = []
        all_lons = []
        for _, row in segments_df.iterrows():
            orig = row['original_real']
            all_lats.extend(orig[:, lat_idx])
            all_lons.extend(orig[:, lon_idx])
            
        if not all_lats: return

        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Initialize Map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='OpenStreetMap')
        
        # Plot Tracks
        for _, row in segments_df.iterrows():
            seg_id = row['segment_id']
            mse = row['mse']
            orig = row['original_real']
            recon = row['recon_real']
            length = row['length'] # Use the actual length
            
            # Folium expects list of (lat, lon) tuples
            # We slice by length to ensure we don't plot padding
            orig_path = list(zip(orig[:length, lat_idx], orig[:length, lon_idx]))
            recon_path = list(zip(recon[:length, lat_idx], recon[:length, lon_idx]))
            
            # 1. Original (Blue Solid Line)
            folium.PolyLine(
                orig_path,
                color='blue',
                weight=3,
                opacity=0.5,
                tooltip=f"ID: {seg_id} (Original)"
            ).add_to(m)
            
            # 2. Reconstructed (Red Dashed Line)
            folium.PolyLine(
                recon_path,
                color='red',
                weight=3,
                opacity=0.7,
                tooltip=f"ID: {seg_id} (Recon) | MSE: {mse:.4f}"
            ).add_to(m)
            
            # 3. Interactive Dots for Reconstructed Points
            for t in range(length):
                p_lat = recon[t, lat_idx]
                p_lon = recon[t, lon_idx]
                
                # Info components
                info_str = f"<b>Step:</b> {t}<br>"
                info_str += f"<b>Lat:</b> {p_lat:.5f}<br><b>Lon:</b> {p_lon:.5f}<br>"
                
                if sog_idx != -1:
                    p_sog = recon[t, sog_idx]
                    info_str += f"<b>SOG:</b> {p_sog:.2f} <br>"
                    
                if has_cog:
                    p_sin = recon[t, cog_sin_idx]
                    p_cos = recon[t, cog_cos_idx]
                    # Convert sin/cos to degrees (0-360)
                    p_cog = (np.degrees(np.arctan2(p_sin, p_cos)) + 360) % 360
                    info_str += f"<b>COG:</b> {p_cog:.1f}°"
                
                # Create a small circle marker for the point
                folium.CircleMarker(
                    location=(p_lat, p_lon),
                    radius=3,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=1.0,
                    popup=folium.Popup(info_str, max_width=200)
                ).add_to(m)

            # 4. Interactive Dots for Original Points
            for t in range(length):
                p_lat = orig[t, lat_idx]
                p_lon = orig[t, lon_idx]
                
                info_str = f"<b>Step:</b> {t}<br>"
                info_str += f"<b>Lat:</b> {p_lat:.5f}<br><b>Lon:</b> {p_lon:.5f}<br>"
                
                if sog_idx != -1:
                    p_sog = orig[t, sog_idx]
                    info_str += f"<b>SOG:</b> {p_sog:.2f} <br>"
                    
                if has_cog:
                    p_sin = orig[t, cog_sin_idx]
                    p_cos = orig[t, cog_cos_idx]
                    p_cog = (np.degrees(np.arctan2(p_sin, p_cos)) + 360) % 360
                    info_str += f"<b>COG:</b> {p_cog:.1f}°"
                
                folium.CircleMarker(
                    location=(p_lat, p_lon),
                    radius=3,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=1.0,
                    popup=folium.Popup(info_str, max_width=200)
                ).add_to(m)

        # Save
        filename = f"{filename_no_ext}.html"
        save_path = os.path.join(self.output_dir, filename)
        m.save(save_path)
        print(f"Map saved: {save_path}")


    
            
       
# ==========================================
# Example Usage
# ==========================================
def testing_run():
    
    # Name of the model configuration to use
    MODEL_NAME = "H256_L64_Lay1_lr0.001_BS64_Drop0.0"
    
    # Data to test on
    PARQUET_FILE = "ais-data/df_preprocessed/pre_processed_df_test_20min_12h_split_03nov.parquet"

    # Output Directory
    OUTPUT_DIR = "test_results/weighted3_20min_12h_split_6months" + "/" + MODEL_NAME
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    WEIGHTS_FILE = "models/weighted3_20min_12h_split_6months" + "/weights_" + MODEL_NAME + ".pth"
    CONFIG_FILE = "models/weighted3_20min_12h_split_6months" + "/config_" + MODEL_NAME + ".json"
    
    # Load Model Config
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    tester = AISTester(config, WEIGHTS_FILE, output_dir=OUTPUT_DIR)
    
    if os.path.exists(PARQUET_FILE):
        # 1. Evaluate ALL data first
        tester.load_data(PARQUET_FILE)
        tester.evaluate()
        
        # 2. Plot General Stats
        tester.plot_error_distributions()
        
        # 3. Plot Filtered Stats (Example)
        # You can pass a list of IDs to filter just the plot without re-running evaluate
        # my_interesting_ids = ["segment_A", "segment_B"]
        # tester.plot_error_distributions(filter_ids=my_interesting_ids, filename_suffix="_special_group")
        
        # 4. Standard Best/Worst
        tester.plot_best_worst_segments(n=3)
        
        # 5. Maps
        tester.generate_maps(n_best_worst=5, n_random=10)

        # 6. Filtered Map Example
        # tester.generate_filtered_map(segment_ids=["segment_1", "segment_2"], map_name="map_special_segments")
        
    else:
        print(f"File {PARQUET_FILE} not found.")
        
        
if __name__ == "__main__":
    testing_run()