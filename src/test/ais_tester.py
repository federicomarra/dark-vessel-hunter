# AIS Tester Class

# File imports
from src.train.ais_dataset import AISDataset, ais_collate_fn
from src.train.model_anchoring import AIS_LSTM_Autoencoder

# Library imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import os
import folium
from folium.features import DivIcon

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

    # ==========================================
    # IMPROVED PLOTTING FUNCTIONS
    # ==========================================

    def plot_error_distributions(self, filter_ids=None, filename_suffix=""):
        """
        Plots the distribution of MSE with percentiles.
        Includes Linear and Log scale views.
        Highlights specific target segments if found.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            return

        # --- Filter Logic ---
        if filter_ids is not None:
            plot_df = self.results_df[self.results_df['segment_id'].isin(filter_ids)]
            title_extra = " (Filtered)"
        else:
            plot_df = self.results_df
            title_extra = ""

        mse_values = plot_df['mse'].values
        
        # --- Target Segments to Highlight ---
        targets_to_mark = [
            "990000001_2025-08-29_0",
            "990000001_2025-08-29_1",
            "990000001_2025-08-29_2",
            "990000002_2025-08-29_0"
        ]
        
        # --- Plot Setup (2 Rows: Linear, Log) ---
        fig, (ax_linear, ax_log) = plt.subplots(2, 1, figsize=(18, 14))
        fig.suptitle(f'Reconstruction Error Distributions (MSE){title_extra}', fontsize=20, fontweight='bold')
        
        # --- 1. Linear Scale Plot ---
        sns.histplot(mse_values, kde=True, ax=ax_linear, color='skyblue', edgecolor='black')
        ax_linear.set_title('Distribution (Linear Scale)', fontsize=16, fontweight='bold')
        ax_linear.set_xlabel('Mean Squared Error')
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        perc_values = np.percentile(mse_values, percentiles)
        colors_p = ['green', 'blue', 'orange', 'red', 'purple']
        
        y_max = ax_linear.get_ylim()[1]
        for i, (p, v) in enumerate(zip(percentiles, perc_values)):
            ax_linear.axvline(v, color=colors_p[i], linestyle='--', alpha=0.8, linewidth=1.5)
            # Add text label
            ax_linear.text(v, y_max * (0.9 - i*0.07), f' {p}th: {v:.5f}', 
                         color=colors_p[i], fontweight='bold', fontsize=11, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # --- 2. Log Scale Plot ---
        sns.histplot(mse_values, kde=True, ax=ax_log, color='salmon', edgecolor='black', log_scale=True)
        ax_log.set_title('Distribution (Log Scale)', fontsize=16, fontweight='bold')
        ax_log.set_xlabel('MSE (Log Scale)')
        
        # Show 90th percentile on log plot for reference
        ax_log.axvline(perc_values[2], color='orange', linestyle='--', label='90th Percentile')
        ax_log.legend()

        # --- 3. Highlight Specific Segments ---
        # Search for segments containing the target strings
        matches = plot_df[plot_df['segment_id'].astype(str).apply(lambda x: any(t in x for t in targets_to_mark))]
        
        if not matches.empty:
            print(f"Found {len(matches)} target segments to mark on the plot.")
            
            # Use a distinct color for the markers
            marker_color = 'magenta'
            
            for _, row in matches.iterrows():
                mse_val = row['mse']
                seg_id = row['segment_id']
                
                # Mark on Linear Plot
                ax_linear.axvline(mse_val, color=marker_color, linestyle='-', linewidth=2.5)
                # Annotation with arrow
                ax_linear.annotate(f"{seg_id}\nMSE: {mse_val:.4f}", 
                                   xy=(mse_val, 0), xycoords=('data', 'axes fraction'),
                                   xytext=(0, 40), textcoords='offset points',
                                   arrowprops=dict(facecolor=marker_color, shrink=0.05),
                                   color=marker_color, fontweight='bold', rotation=45, ha='left')

                # Mark on Log Plot
                ax_log.axvline(mse_val, color=marker_color, linestyle='-', linewidth=2.5)
                ax_log.annotate(f"{seg_id}", 
                                   xy=(mse_val, 0), xycoords=('data', 'axes fraction'),
                                   xytext=(0, 40), textcoords='offset points',
                                   arrowprops=dict(facecolor=marker_color, shrink=0.05),
                                   color=marker_color, fontweight='bold', rotation=45, ha='left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(self.output_dir, f"detailed_distribution{filename_suffix}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Detailed distribution plot saved: {save_path}")

    def plot_best_worst_segments(self, n=3):
        """Standard line plots for best/worst."""
        if not hasattr(self, 'results_df') or self.results_df.empty: return
        sorted_df = self.results_df.sort_values(by='mse')
        
        self._plot_segments_lines(sorted_df.head(n), "BEST")
        self._plot_segments_lines(sorted_df.tail(n), "WORST")

    def _plot_segments_lines(self, segment_df, title_prefix):
        features = self.config['features']
        for _, row in segment_df.iterrows():
            seg_id = row['segment_id']
            mse = row['mse']
            orig, recon = row['original_real'], row['recon_real']
            
            fig, axes = plt.subplots(1, len(features), figsize=(20, 4))
            fig.suptitle(f"{title_prefix}: {seg_id} (MSE: {mse:.5f})", fontsize=14, fontweight='bold')
            
            for i, feature in enumerate(features):
                axes[i].plot(orig[:, i], 'k-', label='Original')
                axes[i].plot(recon[:, i], 'r--', label='Recon')
                axes[i].set_title(feature)
                if i==0: axes[i].legend()
            
            plt.tight_layout()
            safe_id = str(seg_id).replace("/", "_")
            plt.savefig(os.path.join(self.output_dir, f"{title_prefix}_{safe_id}.png"))
            plt.close()

    # ==========================================
    # ADVANCED MAPPING FUNCTIONS
    # ==========================================
    
    def generate_gradient_worst_map(self, n_worst=10):
        """
        Generates a map of the N worst segments.
        Features:
        1. Colored reconstruction scale: Yellow (Best of worst) -> Red (Worst of worst)
        2. Segment ID labels visible on the map.
        """
        if folium is None or not hasattr(self, 'results_df'): return
        
        # Get worst segments
        worst_df = self.results_df.sort_values(by='mse', ascending=False).head(n_worst)
        
        # --- Color Scale Setup ---
        # Normalize MSE values for colormap (0 to 1 range within this group)
        min_mse = worst_df['mse'].min()
        max_mse = worst_df['mse'].max()
        norm = mcolors.Normalize(vmin=min_mse, vmax=max_mse)
        # Use a Red-Yellow colormap (reversed so high error = red)
        cmap = cm.get_cmap('YlOrRd') 
        
        # Calculate Center
        lat_idx = self.config['features'].index('Latitude')
        lon_idx = self.config['features'].index('Longitude')
        
        # Center map on the very worst segment
        center_row = worst_df.iloc[0]
        center_lat = np.mean(center_row['original_real'][:, lat_idx])
        center_lon = np.mean(center_row['original_real'][:, lon_idx])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='OpenStreetMap')
        
        for _, row in worst_df.iterrows():
            orig = row['original_real']
            recon = row['recon_real']
            mse = row['mse']
            seg_id = str(row['segment_id'])
            
            # Determine Color based on MSE
            # If min == max (e.g. n=1), default to red
            if max_mse > min_mse:
                rgba = cmap(norm(mse))
                hex_color = mcolors.to_hex(rgba)
            else:
                hex_color = '#FF0000' # Red
            
            orig_path = list(zip(orig[:, lat_idx], orig[:, lon_idx]))
            recon_path = list(zip(recon[:, lat_idx], recon[:, lon_idx]))
            
            # 1. Plot Original (Grey, subtle)
            folium.PolyLine(
                orig_path, color="#757575", weight=2, opacity=0.8,
                tooltip=f"Original: {seg_id}"
            ).add_to(m)
            
            # 2. Plot Reconstruction (Colored by Error Severity)
            folium.PolyLine(
                recon_path, color=hex_color, weight=4, opacity=0.9,
                tooltip=f"ID: {seg_id} | MSE: {mse:.4f}",
                popup=f"Segment: {seg_id}<br>MSE: {mse:.5f}"
            ).add_to(m)
            
            # 3. Add Text Label for Segment Number
            # We place it at the start of the track
            folium.map.Marker(
                orig_path[0],
                icon=DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 10pt; color: {hex_color}; font-weight: bold;">{seg_id}</div>',
                    )
            ).add_to(m)

            # --- 4. ROI polygon (config.POLYGON_COORDINATES is (lon, lat)) ---
            folium.vector_layers.Polygon(
                locations=[(lat, lon) for lon, lat in config_file.POLYGON_COORDINATES],
                color="black",
                weight=1,
                fill=False,
                tooltip="ROI boundary",
            ).add_to(m)

            # --- 5. Cables (config.CABLE_POINTS has (lat, lon) pairs) ---
            for cable_name, points in config_file.CABLE_POINTS.items():
                # Cable polyline
                folium.PolyLine(
                    locations=[(lat, lon) for lat, lon in points],
                    color="#0000ff",
                    weight=3,
                    opacity=0.9,
                    tooltip=cable_name,
                ).add_to(m)

                # Cable node markers
                for lat, lon in points:
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=3.5,
                        color="#1b1b1b",
                        fill=True,
                        fill_color="white",
                        weight=1,
                    ).add_to(m)
            
        save_path = os.path.join(self.output_dir, f"map_GRADIENT_WORST_{n_worst}.html")
        m.save(save_path)
        print(f"Gradient map saved: {save_path}")

    def generate_maps(self, n_best=5, n_random=5):
        """Standard maps wrapper."""
        if folium is None: return
        sorted_df = self.results_df.sort_values(by='mse')
        
        self._save_html_map(sorted_df.head(n_best), f"map_BEST_{n_best}")
        if len(sorted_df) > n_random:
            self._save_html_map(sorted_df.sample(n=n_random), f"map_RANDOM_{n_random}")

    def _save_html_map(self, df, filename):
        lat_idx = self.config['features'].index('Latitude')
        lon_idx = self.config['features'].index('Longitude')
        
        center_row = df.iloc[0]
        c_lat = np.mean(center_row['original_real'][:, lat_idx])
        c_lon = np.mean(center_row['original_real'][:, lon_idx])
        m = folium.Map(location=[c_lat, c_lon], zoom_start=6)
        
        for _, row in df.iterrows():
            orig = row['original_real']
            recon = row['recon_real']
            orig_path = list(zip(orig[:, lat_idx], orig[:, lon_idx]))
            recon_path = list(zip(recon[:, lat_idx], recon[:, lon_idx]))
            
            folium.PolyLine(orig_path, color='blue', weight=2, opacity=0.5).add_to(m)
            folium.PolyLine(recon_path, color='red', weight=2, opacity=0.5, dash_array='5,5').add_to(m)
            
        m.save(os.path.join(self.output_dir, f"{filename}.html"))
