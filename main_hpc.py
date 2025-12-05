# Main HPC Script to run all steps sequentially

# Run this script to install required packages
# pip install -r requirements.txt

# File imports
from main_1_data import main_data as data
from main_2_preprocess import main_preprocess as preprocess
from main_3_train import main_train as train
from main_3_train_single import main_train_single as train_single
from main_4_test import main_test as test

if __name__ == "__main__":
    # Decide which days to run in config.py
    
    # Run all steps sequentially
    data()
    preprocess()
    #train()
    train_single()
    test()