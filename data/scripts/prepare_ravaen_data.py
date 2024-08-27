"""
`prepare_ravaen_data.py`

since we will not change the patch size very often during the whole project,
we naively save the preprocessed patches in the hdf5 file, and load all patches at once during training,
which enables faster loading and training speed.
"""

# -----------------------------------------------
# Import libraries
from glob import glob
import os
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
# -----------------------------------------------

# -----------------------------------------------
# Define the preprocessed single hdf5 file for the RaVAEn dataset
ravaen_hdf5 = 'data/processed/ravaen_hdf5/ravaen.hdf5'
assert os.path.exists(ravaen_hdf5), f'{ravaen_hdf5} does not exist.'
# -----------------------------------------------

# -----------------------------------------------
# Function to save the patches according to the csv file
def save_patches(hdf5_path, output_hdf5_path, csv_path):
    # Read the csv file
    csv_file = pd.read_csv(csv_path)

    # Load hdf5 file
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        with h5py.File(output_hdf5_path, 'w') as output_hdf5_file:
            for i, eachrow in tqdm(csv_file.iterrows()):
                event = eachrow['event']
                row = eachrow['row']
                col = eachrow['col']

                event_group = hdf5_file[event]
                # Save the patches only in RGB bands (not BGR)
                # output_dim: (3, 64, 64)
                t1 = event_group['t1'][:][[3, 2, 1], row:row+64, col:col+64]
                t2 = event_group['t2'][:][[3, 2, 1], row:row+64, col:col+64]
                t3 = event_group['t3'][:][[3, 2, 1], row:row+64, col:col+64]
                t4 = event_group['t4'][:][[3, 2, 1], row:row+64, col:col+64]
                t5 = event_group['t5'][:][[3, 2, 1], row:row+64, col:col+64]

                # output_dim: (1, 64, 64)
                change_mask = event_group['change'][:][:, row:row+64, col:col+64]

                # Save the patches to the output hdf5 file
                output_group = output_hdf5_file.create_group(f"{event}_{row}_{col}")
                output_group.create_dataset("t1", data=t1)
                output_group.create_dataset("t2", data=t2)
                output_group.create_dataset("t3", data=t3)
                output_group.create_dataset("t4", data=t4)
                output_group.create_dataset("t5", data=t5)
                output_group.create_dataset("change", data=change_mask)

            # Close the hdf5 file
            output_hdf5_file.close()
        hdf5_file.close()
# -----------------------------------------------

# -----------------------------------------------
# Generate the preprocessed training, validation, and test hdf5 files
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        # Define the output hdf5 file
        output_hdf5 = f"data/processed/ravaen_hdf5/{split}_patches.hdf5"

        # Define the csv file
        csv_file = f"data/processed/ravaen_hdf5/{split}.csv"
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'

        # Save the patches according to the csv file
        save_patches(ravaen_hdf5, output_hdf5, csv_file)

        print(f"Saved the preprocessed {split} hdf5 file: {output_hdf5}")

        with h5py.File(output_hdf5, 'r') as hdf:
            print(f"Number of patches in the output hdf5 file: {len(hdf)}")
            # print(f"Keys in the output hdf5 file: {list(hdf.keys())}")
            print(f"Shape of the first patch: {hdf[list(hdf.keys())[0]]['t1'].shape}")
# -----------------------------------------------