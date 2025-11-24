# pip install -r requirements.txt

# Imports
import main_data as data
import main_pre_processing as pre
import main_training as train


if __name__ == "__main__":
    data.main_data()
    pre.main_pre_processing()
    train.main_training()