"""
`prepare_eurosat_data.py`

1. We split the EuroSAT dataset into training, validation and test sets.
2. Since the EuroSAT dataset is small enough to fit into memory,
   we convert the training, validation and test sets into separate HDF5 files for faster loading.
"""

"""
The EuroSAT dataset is a collection of 27000 images of 10 classes.
Each image is 64x64 pixels.

The tree structure of the dataset is as follows:
- EuroSAT
    - AnnualCrop
    - Forest
    - HerbaceousVegetation
    - Highway
    - Industrial
    - Pasture
    - PermanentCrop
    - Residential
    - River
    - SeaLake
"""

# Import libraries
import numpy as np
import h5py
import os
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------------------------
# Define the path to the raw data directory
data_path = 'data/raw/eurosat/2750'

# Define the path to the HDF5 files
hdf5_train_path = 'data/processed/EuroSAT_hdf5/train.hdf5'
hdf5_val_path = 'data/processed/EuroSAT_hdf5/val.hdf5'
hdf5_test_path = 'data/processed/EuroSAT_hdf5/test.hdf5'

if not os.path.exists(os.path.dirname(hdf5_train_path)):
    os.makedirs(os.path.dirname(hdf5_train_path))
# -----------------------------------------------

# -----------------------------------------------
# Get the list of all the subdirectories
subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

# Initialize dicts to hold the split data
train_images = {}
val_images = {}
test_images = {}

# Loop over the subdirectories
for subdir in tqdm(subdirs):
    # Get the list of all the images in this subdirectory
    img_files = [f for f in os.listdir(os.path.join(data_path, subdir)) if f.endswith('.jpg')]

    # Split the image names into training and test sets
    train_imgs, test_imgs = train_test_split(img_files, test_size=0.2, random_state=42)

    # Further split the training images into training and validation sets
    # Here: 0.125 x (1 - 0.2) = 0.1 (10% of the original data)
    # train:val:test = 7:1:2
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.125, random_state=42)

    # Store the split data
    train_images[subdir] = train_imgs
    val_images[subdir] = val_imgs
    test_images[subdir] = test_imgs
# -----------------------------------------------

# -----------------------------------------------
# Function to create HDF5 file for a given set of images
def create_hdf5(img_dict, hdf5_file_path):
    """
    Create HDF5 file for a given set of images.
    
    Args:
        img_dict (dict): Dictionary containing the image names for each class.
        hdf5_file_path (str): Path to the HDF5 file.

    Returns:
        None
    """
    # Initialize the HDF5 file
    hdf5_file = h5py.File(hdf5_file_path, mode='w')
    for subdir, img_files in tqdm(img_dict.items()):
        # Create a group in the file
        hdf5_group = hdf5_file.create_group(subdir)
        for img_file in img_files:
            # Read the image
            img = Image.open(os.path.join(data_path, subdir, img_file))
            # Convert the image to numpy array
            img = np.array(img)
            # Create a dataset in the group
            # hdf5_group.create_dataset(img_file, data=img)

            # Use the image name (without extension) as the dataset name
            img_name, _ = os.path.splitext(img_file)
            hdf5_group.create_dataset(img_name, data=img)

    # Close the HDF5 file
    hdf5_file.close()
# -----------------------------------------------

# -----------------------------------------------
# Create the HDF5 files for training, validation and test sets
print('Creating HDF5 file for training set...')
create_hdf5(train_images, hdf5_train_path)

print('Creating HDF5 file for validation set...')
create_hdf5(val_images, hdf5_val_path)

print('Creating HDF5 file for test set...')
create_hdf5(test_images, hdf5_test_path)
# -----------------------------------------------