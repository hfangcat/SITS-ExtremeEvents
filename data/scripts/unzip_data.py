"""
Unzip the zip files in the data/raw directory.
"""

import os
import zipfile
from tqdm import tqdm

# Define the path to the raw data directory
raw_data_dir = 'data/raw'
assert os.path.exists(raw_data_dir), f'{raw_data_dir} does not exist.'

# Define the output directory
eurosat_dir = 'data/raw/eurosat'
if not os.path.exists(eurosat_dir):
    os.makedirs(eurosat_dir)

ravaen_dir = 'data/raw/ravaen'
if not os.path.exists(ravaen_dir):
    os.makedirs(ravaen_dir)

# Unzip the EuroSAT data
eurosat_zip = 'data/raw/EuroSAT.zip'
assert os.path.exists(eurosat_zip), f'{eurosat_zip} does not exist.'

print('Unzipping EuroSAT data...')

with zipfile.ZipFile(eurosat_zip, 'r') as zip_ref:
    zip_ref.extractall(eurosat_dir)
    zip_ref.close()

print('EuroSAT data unzipped!')

# Unzip the RaVAEn data
zip_names = [
    "fires.zip",
    "floods.zip",
    "hurricanes.zip",
    "landslides.zip"
]

files_to_extract = [os.path.join(raw_data_dir, f) for f in zip_names]

print('Unzipping RaVAEn data...')

for zip_file in files_to_extract:
    assert os.path.exists(zip_file), f'{zip_file} does not exist.'
    print(f'Unzipping {zip_file}...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(ravaen_dir)
        zip_ref.close()

print('RaVAEn data unzipped!')