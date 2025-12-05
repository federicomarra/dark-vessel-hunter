# kattegat-vessel-anomaly-detector
## Project 29 — Group 80  
## Anomaly Detection in AIS Vessel Trajectories Using LSTM Autoencoders

This repository contains the source code for an anomaly detection pipeline designed to identify irregular vessel behaviors (e.g., dark vessels or anomalous movements) near critical infrastructure using AIS (Automatic Identification System) data.  
The core model is an **LSTM Autoencoder** trained to learn normal vessel trajectory patterns and detect deviations.

---

## Project Overview

The pipeline automates the entire workflow — from raw data acquisition to anomaly visualization.  
It is designed for use with **Danish Maritime Authority** AIS data and includes configurable geospatial filtering for a defined Area of Interest (AOI).

### **Key Features**
- **Automated Scraper**: Downloads and unzips AIS data from `aisdata.ais.dk`.
- **Robust Filtering**: Supports bounding box, polygon filters (Denmark/Cable areas), port areas, and vessel class (A/B).
- **Trajectory Segmentation**: Splits raw AIS messages into coherent vessel tracks based on time gaps and duration.
- **Deep Learning Model**: PyTorch implementation of an **LSTM Autoencoder** for trajectory reconstruction.
- **Visualization**: Generates reconstruction-error histograms and interactive HTML maps of best/worst tracks.

---

## Installation & Requirements

### **Prerequisites**
- Python **3.10+**
- **GPU recommended**  
  Training on CPU is possible but slow. CUDA (NVIDIA) or MPS (Apple Silicon) is automatically used if available.

### **Setup**
1. Clone the repository.
2. Install the required Python dependencies:

```bash
pip install -q -r requirements.txt
```

 **Note on Training Speed**  
If you do not have a dedicated GPU, it is highly recommended to lower the `EPOCHS` value (e.g., to 5) in `config.py` to verify functionality without excessive runtimes.

---

##  Usage Pipeline

The main entry point is the Jupyter Notebook **`main.ipynb`**, which executes the workflow in four distinct stages:

---

### 1. Data Download & Filtering

The system:

- Iteratively downloads daily `.csv` AIS files  
- Applies geospatial filters to focus on the Area of Interest  
- Cleanses invalid SOG/COG (Speed/Course Over Ground) values  
- Converts processed data into **Parquet** format for efficient reading  

**Source:**  
`src/data/ais_downloader`, `src/data/ais_filtering`  

**Output directory:**  
`ais-data/parquet/`

---

### 2. Preprocessing

Loads the Parquet files and prepares the dataset for training the LSTM Autoencoder.

#### Steps:
- **Feature Engineering:** Converts COG to sine/cosine components  
- **Grouping:** Aggregates ship types (Commercial, Passenger, Service, Fishing)  
- **Resampling:** Normalizes AIS message timestamps to fixed time intervals  
- **Train/Test Split:** Example → Train (Aug 1–7), Test (Aug 8)

**Output directory:**  
`ais-data/df_preprocessed/`

---

### 3. Training

Trains the LSTM Autoencoder to learn vessel trajectory patterns and reconstruct them.

#### Model Overview:
- **Architecture:** Configurable hidden dimensions, latent size, and number of LSTM layers  
- **Loss Function:** Mean Squared Error (MSE)  
- **Early Stopping:** Triggered using validation loss  

#### Outputs:
- Model weights → `models/`  
- Training logs → `train_output/`

---

### 4. Testing & Evaluation

The `AISTester` class loads the trained model and evaluates anomaly scores on the test set.

#### Features:
- **Anomaly Detection:** Tracks with highest reconstruction error are flagged  
- **Visualization Tools:**  
  - **Error Histograms:** Distribution of reconstruction errors  
  - **Interactive Maps:** HTML files showing  
    - *Best (normal) trajectories*  
    - *Worst (anomalous) trajectories*

#### Output directory:
`test_results/`

---

##  Project Structure

```plaintext
├── main_for_professor.ipynb   # Main execution notebook
├── config.py                  # Global configuration (BBOX, Hyperparameters)
├── requirements.txt           # Python dependencies
├── src/
│   ├── data/                  # Scrapers, readers, and filters
│   ├── pre_proc/              # Segmentation and resampling logic
│   ├── train/                 # Model definition (LSTM AE) and training loop
│   └── test/                  # Inference and visualization tools
├── ais-data/                  # Storage for raw CSV, Parquet, and processed data
├── models/                    # Saved model weights
└── test_results/              # Generated plots and HTML maps
```


##  Authors

**Group 80 — DTU Deep Learning, Project 29**
- Federico Marra (https://github.com/federicomarra)
- Giacomo Stefanizzi (https://github.com/jackstefa)
- Luca Saladino (https://github.com/Lucasala2003)
- Eleonora Failla (https://github.com/elefaill)
- Flavio Sarno (https://github.com/Flavioonras)