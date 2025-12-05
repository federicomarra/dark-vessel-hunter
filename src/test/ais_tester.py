# AIS Tester Class

# File imports
from src.train.ais_dataset import AISDataset, ais_collate_fn
from src.train.model import AIS_LSTM_Autoencoder

# Library imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import folium

import config as config_file
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
        # Store config
        self.config = model_config
        self.loss_type = model_config.get('loss_type', 'mse') 
        self.use_weighted_loss_metric = (self.loss_type == 'weighted_mse')
        
        
        # Device setup
        if torch.cuda.is_available():
            actual_device = torch.device("cuda")  # for PC with NVIDIA
        elif torch.backends.mps.is_available():
            actual_device = torch.device("mps")   # for Mac Apple Silicon
        else:
            actual_device = torch.device("cpu")   # Fallback on CPU
        
        self.device = actual_device if device is None else device
        print(f"Using device: {self.device}")
        
        # Output directory setup
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
        use_weighted_loss = self.use_weighted_loss_metric

        if use_weighted_loss:
            # order [Lat, Lon, SOG, COG_sin, COG_cos]
            feature_weights = torch.tensor(
                [1.0, 1.0, 2.0, 2.0, 2.0], 
                dtype=torch.float32, 
                device=self.device 
            ) 
            weights_expanded = feature_weights.view(1, 1, -1)

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
                
                if use_weighted_loss:
                    weighted_errors = raw_errors * weights_expanded
                else:
                    weighted_errors = raw_errors 
                    
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
                    errors_for_seg = weighted_errors[i, :length, :]
                    error_per_feat_mean = errors_for_seg.mean(dim=0).cpu().numpy()
                    total_mse = errors_for_seg.mean().item()
                    
                    # Inverse Transform to get real units
                    original = padded_seqs[i, :length, :].cpu().numpy()
                    recon = reconstructed[i, :length, :].cpu().numpy()
                    original_real = self.dataset.scaler.inverse_transform(original)
                    recon_real = self.dataset.scaler.inverse_transform(recon)
                    
                    results.append({
                        'segment_id': seg_id,
                        'mse': total_mse,
                        'mse_per_feature': error_per_feat_mean,
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
        
        if hasattr(config_file, 'CABLE_POINTS'):
            for cable_name, points in config_file.CABLE_POINTS.items():
                folium.PolyLine(
                    locations=points,
                    color='green',
                    weight=2,       # Very thin
                    opacity=0.8,
                    tooltip=cable_name  # Show name when hovering
                ).add_to(m)
        
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
