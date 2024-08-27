# SITS-ExtremeEvents
Code and models for extreme event detection using satellite image time series (SITS).

## 1. Data Preparation

**Important Note on Data Preparation and Reproducility:**

We recommend downloading the processed data directly from the provided links in Section 1.1 for the following reasons:
1. Time efficiency: Preparing the data from scratch can take a significant amount of time (e.g., processing RaVAEn data takes over 7 hours).
2. Consistency and Reproducibility: We inadvertently forgot to set a fixed random seed for the data splitting process. As a result, the exact train/validation/test splits may vary slightly if you choose to prepare the data from scratch using the provided scripts. 

For full transparency, we also provide the scripts to prepare the data from scratch in Section 1.2. However, to ensure consistency and reproducibility of the experiments and results as presented in the paper, we strongly suggest using the processed data files, which include the exact splits, available for download in Section 1.1.

### 1.1. Downloading the data
To run the experiments and reproduce the results, you need to download the processed data from the following links and place them in the `data/processed/` directory:
<!-- TODO: Add Google Drive links before publication -->
- EuroSAT (hdf5 format, including train/validation/test splits): [Google Drive]()
- RaVAEn (hdf5 format, including train/validation/test splits): [Google Drive]()

### 1.2. Preparing the data from scratch (optional)
If you want to prepare the data from scratch, you can follow the instructions here:
1. Download the raw data from the following links and place them in the `data/raw/` directory:
    - EuroSAT (RGB): [GitHub](https://github.com/phelber/EuroSAT), [Download Link](https://madm.dfki.de/files/sentinel/EuroSAT.zip)
    - RaVAEn: 
        - Manually download from the RaVAEn paper website: [Google Drive](https://drive.google.com/drive/folders/1VEf49IDYFXGKcfvMsfh33VSiyx5MpHEn)
        - Or alternatively download the data with gdown:
        ```python
        #!gdown https://drive.google.com/uc?id=1UCNnxaL9pQSkkZQx0aDEWQL0UBXPXkv0 -O fires.zip
        #!gdown https://drive.google.com/uc?id=1CbNGrpK66Hos_TtOEut510k7CSHvSwkl -O landslides.zip
        #!gdown https://drive.google.com/uc?id=1VP3SYgh3bj6uPa4r_bKP-5zFP3JdGin8 -O hurricanes.zip
        #!gdown https://drive.google.com/uc?id=1scjd4gIB_eiNS-CsOyb7Q8rYWnl9TM-L -O floods.zip
        ```

2. Unzip the downloaded files:
    ```bash
    # Make sure you are in the 'SITS-ExtremeEvents/' directory
    python data/scripts/unzip_data.py
    ```

3. Run the processing scripts to prepare the data:
    - EuroSAT:
    ```bash
    # Make sure you are in the 'SITS-ExtremeEvents/' directory
    python data/scripts/prepare_eurosat_data.py
    ```
    - RaVAEn:
    ```bash
    # Make sure you are in the 'SITS-ExtremeEvents/' directory
    python data/scripts/ravaen_to_hdf5.py
    python data/scripts/generate_ravaen_splits.py
    python data/scripts/prepare_ravaen_data.py
    ```

<!-- **Important Note:** In the initial setup of this project, we forgot to set a fixed random seed for the data splitting process. As a result, the exact train/validation/test splits may vary slightly if you choose to prepare the data from scratch using the provided scripts (as outlined above in Section 1.2).

To ensure reproducility of the experiments and results as presented in the paper, we have provided the exact train/validation/test splits in the processed data files that you can download from the links in Section 1.1. -->

## 2. Experiments
### 2.1. Training the models

### 2.2. Evaluating the models

### 2.3 Model Checkpoints to reproduce the results (optional)