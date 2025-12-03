# Configuration file for Dark Vessel Hunter project

# ---------------------------------------
# DYNAMIC CONFIGURATION VALUES BELOW
# ---------------------------------------

# ---- DATA CONFIGURATION ----
VERBOSE_MODE = True

START_DATE = "2025-08-01"  # Start date for data downloading
END_DATE   = "2025-08-30"  # End date for data downloading

AIS_DATA_FOLDER = "ais-data" # Root folder to store AIS data
DELETE_DOWNLOADED_ZIP = True  # Whether to delete downloaded zip files after extraction
DELETE_DOWNLOADED_CSV = False # Whether to delete downloaded CSV files after processing

# ---- PRE-PROCESSING CONFIGURATION ----
TRAIN_START_DATE = "2025-08-01"
TRAIN_END_DATE = "2025-08-28"

TEST_START_DATE = "2025-08-29"
TEST_END_DATE = "2025-08-30"

MAX_TIME_GAP_SEC = 15 * 60              # 15 minutes in seconds
MAX_TRACK_DURATION_SEC = 12 * 60 * 60   # 12 hours in seconds
MIN_TRACK_DURATION_SEC = 10 * 60        # 10 minutes in seconds
MIN_SEGMENT_LENGTH = None               # 10 # datapoints
MIN_FREQ_POINTS_PER_MIN = 1          # Minimum frequency of points per minute in a segment
RESAMPLING_RULE = "1min"  # Resampling rule for time series data

SEGMENT_MAX_LENGTH = 300  # datapoints TODO: non sappiamo cos'è, jack lo sai??

# ---- TRAINING CONFIGURATION ---- # TODO: da sistemare
BATCH_SIZE = 128
EPOCHS = 30
HIDDEN_DIM = 64
LATENT_DIM = 16
NUM_LAYERS = 2
SHIPTYPE_EMB_DIM = 8
LEARNING_RATE = 1e-3
BETA = 1e-3







# ---------------------------------------
# STATIC CONFIGURATION VALUES BELOW
# ---------------------------------------

#  ---- DATA FILTERING CONFIGURATION ----
VESSEL_AIS_CLASS = ("Class A", "Class B")

# ---- SOG FILTERING CONFIGURATION ----
REMOVE_ZERO_SOG_VESSELS = False # Whether to remove vessels with zero Speed Over Ground
SOG_IN_MS = True                # If True, SOG is in meters/second; if False, SOG is in knots
SOG_MIN_KNOTS = 0.5             # Minimum SOG in knots
SOG_MAX_KNOTS = 35.0            # Maximum SOG in knots


# Bounding Box to prefilter AIS data [lat_max, lon_min, lat_min, lon_max]
BBOX = [57.58, 10.5, 57.12, 11.92]

# Polygon coordinates for precise Area of Interest (AOI) filtering (lon, lat)
POLYGON_COORDINATES = [
    (10.5162, 57.3500),  # coast top left (lon, lat)
    (10.9314, 57.5120),  # sea top left
    (11.5128, 57.5785),  # sea top right
    (11.9132, 57.5230),  # top right (Swedish coast)
    (11.9189, 57.4078),  # bottom right (Swedish coast)
    (11.2133, 57.1389),  # sea bottom right
    (11.0067, 57.1352),  # sea bottom left
    (10.5400, 57.1880),  # coast bottom left
    (10.5162, 57.3500),  # close polygon
]

cable_CG2_points = [(57.30158, 10.53598), (57.30987, 10.56213), (57.32877, 10.62335), (57.44588,10.95990), (57.48487, 11.27270), (57.51543, 11.50590) ,(57.47733, 11.74353), (57.46635, 11.82507), (57.45608, 11.91480)]
cable_kattegat2A_points = [(57.23810, 10.54834), (57.25287, 10.56423), (57.26348,10.65152), (57.25885, 10.73908),(57.26067, 10.75360), (57.25628, 10.79103), (57.25233, 10.86909)]
cable_kattegat2B_points = [(57.30917, 11.19625), (57.39863, 11.48437), (57.44917,11.64585), (57.46658, 11.76620), (57.46273, 11.77677), (57.46470, 11.82323), (57.46743, 11.85143), (57.46640, 11.89138), (57.45608, 11.91480)]

CABLE_POINTS = {
    "CG2": cable_CG2_points,
    "Kattegat2A": cable_kattegat2A_points,
    "Kattegat2B": cable_kattegat2B_points,
}

PRE_PROCESSING_DF_TRAIN_PATH = "ais-data/df_preprocessed/pre_processed_df_train.parquet"
PRE_PROCESSING_DF_TEST_PATH = "ais-data/df_preprocessed/pre_processed_df_test.parquet"
PRE_PROCESSING_METADATA_TRAIN_PATH = "ais-data/df_preprocessed/pre_processing_metadata_train.json"
PRE_PROCESSING_METADATA_TEST_PATH = "ais-data/df_preprocessed/pre_processing_metadata_test.json"
RAW_PARQUET_ROOT = "ais-data/parquet"
TEST_OUTPUT_CSV = f"runs/test_{TEST_START_DATE}_{TEST_END_DATE}_scores.csv"

NUMERIC_COLS = [   # Columns to be normalized
    "Latitude", 
    "Longitude",
    "SOG",
    "COG_sin",
    "COG_cos",
]

SHIPTYPE_TO_ID = {
    "Commercial": 0,
    "Fishing": 1,
    "Other": 2,
    "Passenger": 3,
    "Service": 4,
}

ID_TO_SHIPTYPE = {
    0: "Commercial",
    1: "Fishing",
    2: "Other",
    3: "Passenger",
    4: "Service",
}


# NAV_ONEHOT_COLS = [
#     'NavStatus_0',
#     'NavStatus_1',
#     'NavStatus_2',
#     'NavStatus_3',
#     'NavStatus_4',
#     'NavStatus_5',
#     'NavStatus_6'
# ]

FEATURE_COLS = NUMERIC_COLS #+ NAV_ONEHOT_COLS


# ---- MODEL EVALUATION ----
WEIGHTS_PATH = "models/AE_simple.pth"
PLOT_PATH = "eval/plots"
PREDICTION_DF_PATH = "eval/predictions_df.parquet"
PREDICTION_DENORM_DF_PATH = "eval/predictions_denorm_df.parquet"
MAPS_PATH = "eval/maps"


FEATURE_INDICES = [0, 1, 2, 3, 4]
FEATURE_NAMES = {
    0: "Latitude",
    1: "Longitude",
    2: "SOG",
    3: "COG_sin",
    4: "COG_cos",
}