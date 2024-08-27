"""
`ravaen_to_hdf5.py`: Convert the RaVAEn dataset to an HDF5 file.
"""

from glob import glob
import os
from tqdm import tqdm
import rasterio
import h5py
import numpy as np

# -----------------------------------------------
# Define the path to the raw data directory
ravaen_folder = 'data/raw/ravaen'
assert os.path.exists(ravaen_folder), f'{ravaen_folder} does not exist.'
events = glob(os.path.join(os.path.join(ravaen_folder, "*"), "*"))
print(f"Found {len(events)} events in the ravaen dataset.")

# Define the output hdf5 file
output_hdf5 = 'data/processed/ravaen_hdf5/ravaen.hdf5'
if not os.path.exists(os.path.dirname(output_hdf5)):
    os.makedirs(os.path.dirname(output_hdf5))
# -----------------------------------------------

# -----------------------------------------------
# Create the output hdf5 file
with h5py.File(output_hdf5, 'w') as hdf:
    for event in tqdm(events):
        event_name = os.path.basename(event)
        
        if os.path.isdir(event):
            print(f"Processing event: {event_name}")
            event_group = hdf.create_group(event_name)

            # read and save the S2 tif files
            s2_tif_files = sorted(glob(os.path.join(event, "S2", "*.tif")))
            assert len(s2_tif_files) == 5, f"Expecting 5 S2 tif files, but got {len(s2_tif_files)}!"
            for i, f in enumerate(s2_tif_files):
                with rasterio.open(f) as src:
                    data = src.read()
                    data = np.nan_to_num(data, copy=True).astype(np.float32)
                    event_group.create_dataset(f"t{i+1}", data=data)
                    # save the metadata
                    event_group[f"t{i+1}"].attrs['date'] = os.path.basename(f).split(".")[0]
            # check if t1, t2, t3, t4, t5 are in the ascending order
            dates = [event_group[f"t{i+1}"].attrs['date'] for i in range(5)]
            assert dates == sorted(dates), f"Dates are not in the ascending order: {dates}"

            # read and save the change map tif file
            change_tif_file = glob(os.path.join(event, "changes", "*.tif"))
            assert len(change_tif_file) == 1, f"Expecting 1 change map tif file, but got {len(change_tif_file)}!"
            with rasterio.open(change_tif_file[0]) as src:
                data = src.read([1])
                event_group.create_dataset("change", data=data)
                # save the metadata
                event_group["change"].attrs['date'] = os.path.basename(change_tif_file[0]).split(".")[0]
# -----------------------------------------------