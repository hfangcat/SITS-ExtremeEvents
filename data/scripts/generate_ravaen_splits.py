"""
`generate_ravaen_splits.py`: Split the RaVAEn dataset into training, validation, and test sets.

We use the same split ratio and the same patch size as the EuroSAT dataset.
(split ratio -> train:val:test = 7:1:2, patch size -> 64x64)

The script performs the following steps:
1. Load the RaVAEn dataset from the generated single hdf5 file.
2. Cut into patches and save the row, col etc. into a single csv file.
3. Split the csv file into training, validation, and test csv files.
"""

# -----------------------------------------------
# Import libraries
from glob import glob
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
# -----------------------------------------------

# -----------------------------------------------
def split_single_csv(h5_file, csv_file, patch_size=64, stride=64):
    """
    Split the ravaen dataset into patches and only save the row, col, and event name into a single csv file.

    Args:
    - h5_file: str, the path to the ravaen hdf5 file.
    - csv_file: str, the path to the output csv file.
    - patch_size: int, the size of the patch.
    - stride: int, the stride of the patch.
    """
    # Load the ravaen dataset
    with h5py.File(h5_file, 'r') as hdf:
        events = list(hdf.keys())
        print(f"Found {len(events)} events in the ravaen dataset.")

        # Create the output csv file
        f = pd.DataFrame(columns=["event", "row", "col"])

        # Loop over each event
        for event in tqdm(events):
            event_group = hdf[event]
            t1 = event_group['t1'][:]
            t2 = event_group['t2'][:]
            t3 = event_group['t3'][:]
            t4 = event_group['t4'][:]
            t5 = event_group['t5'][:]
            change = event_group['change'][:]

            # check if t1, t2, t3, t4, t5, change are in the same shape
            assert t1.shape[1:] == t2.shape[1:] == t3.shape[1:] == t4.shape[1:] == t5.shape[1:] == change.shape[1:], \
            "Shapes of t1, t2, t3, t4, t5, change are not the same!"

            # Split into patches
            print(f"Splitting image into patches for event: {event}")
            _, height, width = t1.shape

            patch_num = 0
            # Skip the last patch if the patch size is not divisible by the stride
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    # we only save the left-top corner of the patch
                    row = pd.Series({"event": event, "row": i, "col": j})
                    f = f.append(row, ignore_index=True)

                    patch_num += 1

            print(f"There are {patch_num} patches in total for event: {event}")

        # Save the csv file
        f.to_csv(csv_file, index=False)
# -----------------------------------------------

# -----------------------------------------------
# Save the output csv file for the whole dataset
# Split the csv file into training, validation, and test csv files
if __name__ == "__main__":
    # Define the input hdf5 file
    ravaen_hdf5 = 'data/processed/ravaen_hdf5/ravaen.hdf5'
    assert os.path.exists(ravaen_hdf5), f'{ravaen_hdf5} does not exist.'

    # Define the output csv file
    output_csv = 'data/processed/ravaen_hdf5/ravaen_patches.csv'

    # Split the ravaen dataset into patches and save the row, col, and event name into a single csv file
    split_single_csv(ravaen_hdf5, output_csv)
    print(f"Saved the output csv file for the whole dataset: {output_csv}")

    # Split the csv file into training, validation, and test csv files
    print("Splitting the csv file into training, validation, and test csv files...")

    df = pd.read_csv(output_csv)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
 
    train.to_csv('data/processed/ravaen_hdf5/train.csv', index=False)
    val.to_csv('data/processed/ravaen_hdf5/val.csv', index=False)
    test.to_csv('data/processed/ravaen_hdf5/test.csv', index=False)

    print("Saved the training, validation, and test csv files.")
# -----------------------------------------------