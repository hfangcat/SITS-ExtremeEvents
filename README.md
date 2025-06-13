# SITS-ExtremeEvents
Official Implementation (code and models) of our WACVw paper: "Leveraging Satellite Image Time Series for Accurate Extreme Event Detection".

[![Paper](https://img.shields.io/badge/Paper-WACVw%202025-blue)](https://openaccess.thecvf.com/content/WACV2025W/GeoCV/html/Fang_Leveraging_Satellite_Image_Time_Series_for_Accurate_Extreme_Event_Detection_WACVW_2025_paper.html)
[![Google Drive](https://img.shields.io/badge/Data-Google%20Drive-orange)](https://drive.google.com/drive/folders/13xS_ewTuenqaj7yCZ9g4_2RNCcQNDCMU?usp=drive_link)

## Overview

SITS-Extreme is a scalable framework for detecting extreme events (e.g., floods, fires, landslides, hurricanes) by leveraging multi-temporal satellite imagery. Unlike traditional bi-temporal approaches, our method integrates multiple pre-disaster images to isolate disaster-relevant changes and filter out irrelevant variations (e.g., weather, seasonality).

<p align="center">
  <img src="figures/teaser.png" width="700"/><br>
  <em>Figure 1: Real-world illustration of SITS-Extreme in detecting extreme events.</em>
</p>

<br/>

<p align="center">
  <img src="figures/pipeline.png" width="700"/><br>
  <em>Figure 2: Overview of the SITS-Extreme pipeline.</em>
</p>




## 0. Environment Setup 🔧

We recommend using **conda** to manage the environment and **pip** to install the required packages.

This project uses **Python 3.8** and was tested with **PyTorch 1.12.1 + CUDA 11.6**.

Create a new conda environment:
```bash
conda create -n sits-extreme python=3.8 -y
conda activate sits-extreme
```

Install the required packages via pip:
```bash
pip install -r requirements.txt
```

⚠️ Notes:
- The PyTorch versions in requirements.txt are CUDA-specific (+cu116). If you're using a different CUDA version or a CPU-only setup, install PyTorch manually before installing other dependencies: Visit https://pytorch.org/get-started/locally/ to get the right install command.

## 1. Data Preparation

**Important Note on Data Preparation and Reproducility:**

We recommend downloading the processed data directly from the provided links in Section 1.1 for the following reasons:
1. Time efficiency: Preparing the data from scratch can take a significant amount of time (e.g., processing RaVAEn data takes over 7 hours).
2. Consistency and Reproducibility: We inadvertently forgot to set a fixed random seed for the data splitting process. As a result, the exact train/validation/test splits may vary slightly if you choose to prepare the data from scratch using the provided scripts. 

For full transparency, we also provide the scripts to prepare the data from scratch in Section 1.2. However, to ensure consistency and reproducibility of the experiments and results as presented in the paper, we strongly suggest using the processed data files, which include the exact splits, available for download in Section 1.1.

### 1.1. Downloading the data
To run the experiments and reproduce the results, you need to download the processed data from the following links and place them in the `data/processed/` directory:
<!-- TODO: Add Google Drive links before publication -->
- EuroSAT (hdf5 format, including train/validation/test splits): [Google Drive](https://drive.google.com/drive/folders/1f15oaI0-6Qqw-R25pXq-nTLN9ITS0VQp?usp=drive_link)
- RaVAEn (hdf5 format, including train/validation/test splits): [Google Drive](https://drive.google.com/drive/folders/1b5020p-2uq7RB-SNqIudJcqTii49KYKa?usp=drive_link)

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
```bash
python main.py --config ${CONFIG_FILE} --subconfig ${SUBCONFIG_FILE} --seed ${SEED}
```

For example, to train our model (SITS-Extreme-VAE) on the RaVAEn dataset with the default configuration and a specific seed, you can run:
```bash
python main.py --config "configs/ravaen/default.yaml" --subconfig "configs/ravaen/vae.yaml" --seed 42
```

### 2.2. Evaluating the models (calculating metrics for all seeds and aggregating results)
```bash
python make_table.py --default_config ${CONFIG_FILE} --subconfig ${SUBCONFIG_FILE} --checkpoint_folder ${CHECKPOINT_FOLDER} --dataset ${DATASET}
```

For example, to evaluate our model (SITS-Extreme-VAE) on the RaVAEn dataset with the default configuration and a specific checkpoint folder, you can run:
```bash
python make_table.py --default_config "configs/ravaen/default.yaml" --subconfig $"configs/ravaen/vae.yaml" --checkpoint_folder "checkpoints/ravaen" --dataset "ravaen"
```

### 2.3 Model Checkpoints to reproduce the results (optional)
We provide the model checkpoints (SITS-Extreme-VAE) on the RaVAEn dataset to reproduce the results in the paper. You can create a folder named `checkpoints/ravaen`, download the model checkpoints from the following link and place them in the `checkpoints/ravaen` directory:
- RaVAEn (SITS-Extreme-VAE): [Google Drive](https://drive.google.com/drive/folders/1om-kO4G-CBONe4-er_j1hX1xEK8CQiF5?usp=drive_link) 


## TODOs
- [x] add environment setup
- [x] add Google Drive links for downloading the processed data
- [x] add example commands for training and testing
- [x] add model checkpoints for reproducing the results
- [x] add reference bibtex
- [ ] Test the code from scratch and fix any bugs
- [x] add license file
- [x] add README improvements (e.g., one or two sentences for project summary, arxiv/thecvf link, overview figure, etc.)
- [ ] add sample output or result logs (optional)


## Reference
If you want to cite our work, you can do so with the following BibTex:
```bibtex
@inproceedings{fang2025leveraging,
  title={Leveraging Satellite Image Time Series for Accurate Extreme Event Detection},
  author={Fang, Heng and Azizpour, Hossein},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision},
  pages={526--535},
  year={2025}
}
```

## Contact
If you have any questions or need further assistance, please feel free to reach out to us: hfang@kth.se.

## Acknowledgements
1. We would like to thank the authors of the RaVAEn paper for providing the RaVAEn dataset and the authors of the EuroSAT dataset for making it publicly available.
2. This work is funded by Digital Futures in the project EO-AI4GlobalChange. All experiments were performed using the supercomputing resource Berzelius provided by the National Supercomputer Centre at Linkoping University and the Knut and Alice Wallenberg Foundation. Heng Fang thanks Erik Englesson, Adam Stewart, Dino Ienco, Zhuo Zheng, Sebastian Gerard, Ling Li, and Sebastian Hafner for their feedback on improving the presentation of this paper.