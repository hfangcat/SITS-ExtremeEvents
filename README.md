# SITS-ExtremeEvents
Official Implementation (code and models) of our WACVw paper: "Leveraging Satellite Image Time Series for Accurate Extreme Event Detection"

## Abstract
Climate change is leading to an increase in extreme weather events causing significant environmental damage and loss of life. Early detection of such events is essential for improving disaster response. In this work we propose SITS-Extreme a novel framework that leverages satellite image time series to detect extreme events by incorporating multiple pre-disaster observations. This approach effectively filters out irrelevant changes while isolating disaster-relevant signals enabling more accurate detection. Extensive experiments on both real-world and synthetic datasets validate the effectiveness of SITS-Extreme demonstrating substantial improvements over widely used strong bi-temporal baselines. Additionally we examine the impact of incorporating more timesteps analyze the contribution of key components in our framework and evaluate its performance across different disaster types offering valuable insights into its scalability and applicability for large-scale disaster monitoring.

## Features
- [x] Implements the SITS-Extreme framework for extreme event detection using satellite image time series.
- [x] Supports training and evaluation from scratch.
- [x] Includes processed dataset and pre-trained model checkpoints.
- [x] Easy-to-use command-line interface and modular structure for flexibility and extensibility.

## TODOs
- [ ] add environment setup
- [x] add Google Drive links for downloading the processed data
- [ ] add example commands for training and testing
- [ ] add model checkpoints for reproducing the results
- [x] add reference bibtex
- [ ] Test the code from scratch and fix any bugs
- [x] add license file
- [ ] add README improvements (e.g., one or two sentences for project summary, arxiv/thecvf link, overview figure, etc.)
- [ ] add sample output or result logs (optional)

## 0. Environment Setup

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
### 2.2. Evaluating the models
```bash
python test.py --default_config ${CONFIG_FILE} --subconfig ${SUBCONFIG_FILE} --checkpoint_folder ${CHECKPOINT_FOLDER}
```

### 2.3 Model Checkpoints to reproduce the results (optional)

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