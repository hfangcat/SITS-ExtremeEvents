"""
Dataset class for EuroSAT dataset (hdf5 version).

For training, validation and testing, we need to generate a time series of 5 images for each sample.
t1 - t4: base image (with irrelevant changes).
t5: base image (with relevant and irrelevant changes).
"""

"""
Training: irrelevant (t1 - t5) + relevant (t5) + same augmentations (t1 - t5) + z-score normalization
Validation & Testing: irrelevant (t1 - t5) + relevant (t5) + z-score normalization
"""

# Import libraries
import os
import numpy as np
import random

# comment out the following line when we want to set the seed randomly
np.random.seed(42)
random.seed(42)

import torch
from tqdm import tqdm
import h5py
from PIL import Image, ImageDraw, ImageEnhance
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .utils import gauss_noise, saturation, brightness, contrast


class EuroSATHDF5Dataset(Dataset):
    """
    Dataset class for EuroSAT dataset (hdf5 version).

    Args:
        hdf5_file_path (str): Path to the hdf5 file.
        config (dict): Dictionary containing the configuration parameters.
                       In this case, it contains the following keys:
                       Irrelevant changes:
                       - brightness: Brightness factor for ColorJitter.
                       - contrast: Contrast factor for ColorJitter.
                       - saturation: Saturation factor for ColorJitter.
                       - hue: Hue factor for ColorJitter.
                       - seasonal_change_prob: Probability of seasonal changes. (default: 0.8)
                       - cloud_cover_prob: Probability of cloud cover. (default: 0.2)
                       Relevant changes:
                       - type: intra-class, inter-class or both.
                       - prob: Probability of relevant changes. (default: 0.5)
                       Normalization:
                       - mean: Mean of the dataset.
                       - std: Standard deviation of the dataset.

    Returns:
        sample (dict): Dictionary containing the following keys:
                       - t1, t2, t3, t4, t5: Images.
                       - label: change (1) or no-change (0).
    """

    # Since EuroSAT is a small dataset, we can load all the images in memory
    def __init__(self, hdf5_file_path, mode, config):
        super().__init__()
        self.hdf5_file_path = hdf5_file_path
        self.config = config
        self.mode = mode

        self.image = []
        self.label = []

        # Read the HDF5 file
        with h5py.File(self.hdf5_file_path, 'r') as f:
            for group in f.keys():
                for img in f[group]:
                    self.image.append(f[group][img][:])
                    self.label.append(group)


    def __len__(self):
        return len(self.image)


    def __getitem__(self, idx):
        """ Get the base image and label """
        base_img = self.image[idx]
        label = self.label[idx]

        # Make five copies of the base image for the time series (t1 - t5)
        t1 = base_img.copy()
        t2 = base_img.copy()
        t3 = base_img.copy()
        t4 = base_img.copy()
        t5 = base_img.copy()

        img_seq = [t1, t2, t3, t4, t5]

        """ Apply seasonal change and cloud cover transforms to t1 - t4 """
        # For t1 - t4, we apply irrelevant changes randomly

        # Define seasonal change transform
        # seasonal_change_transform = transforms.ColorJitter(
        #     brightness=self.config['irrelevant_changes']['brightness'],
        #     contrast=self.config['irrelevant_changes']['contrast'],
        #     saturation=self.config['irrelevant_changes']['saturation'],
        #     hue=self.config['irrelevant_changes']['hue']
        # )

        # 2023/10/23: We replace default ColorJitter with CustomColorJitter
        seasonal_change_transform = CustomColorJitter(
            brightness=self.config['datasets']['irrelevant_changes']['brightness'],
            contrast=self.config['datasets']['irrelevant_changes']['contrast'],
            saturation=self.config['datasets']['irrelevant_changes']['saturation'],
            hue=self.config['datasets']['irrelevant_changes']['hue'],
            prob=self.config['datasets']['irrelevant_changes']['seasonal_change_prob']
        )

        # Define cloud cover transform
        cloud_cover_transform = transforms.Lambda(self.add_cloud)

        # Convert t1 - t4 to PIL images (since ColorJitter and add_cloud work with PIL images)
        for i in range(len(img_seq) - 1):
            img_seq[i] = Image.fromarray(img_seq[i])

        for i in range(len(img_seq) - 1):
            # if random.random() < self.config['irrelevant_changes']['seasonal_change_prob']:
            #     img_seq[i] = seasonal_change_transform(img_seq[i])

            # 2023/10/23: We replace default ColorJitter with CustomColorJitter
            img_seq[i] = seasonal_change_transform(img_seq[i])

            if random.random() < self.config['datasets']['irrelevant_changes']['cloud_cover_prob']:
                img_seq[i] = cloud_cover_transform(img_seq[i])

            # Convert t1 - t4 to numpy arrays
            img_seq[i] = np.array(img_seq[i])

        """ Apply relevant changes (and irrelevant changes) to the t5 image """
        # For t5, we apply relevant changes (CutMix -> intra-class, inter-class or both) randomly
        
        # Implement CutMix
        # Choose a random image from the dataset (intra-class or inter-class or both)
        change = 0

        if random.random() < self.config['datasets']['relevant_changes']['prob']:
            if self.config['datasets']['relevant_changes']['type'] == 'intra-class':
                rand_index = random.choice([i for i in range(len(self)) if self.label[i] == label])
            elif self.config['datasets']['relevant_changes']['type'] == 'inter-class':
                rand_index = random.choice([i for i in range(len(self)) if self.label[i] != label])
            else:
                rand_index = random.choice([i for i in range(len(self))])

            # Get the image and label
            img_sub, label_sub = self.image[rand_index], self.label[rand_index]

            # Generate a random mask for CutMix (0: base image, 1: random image)
            # 2023/10/23: set Gaussian filter as a hyper-parameter
            mask = self.generate_soft_mask(img_seq[4].shape, apply_gaussian=self.config['datasets']['relevant_changes']['apply_gaussian'])
            # Add extra dimension to the mask (for broadcasting) -> (H, W) -> (H, W, 1)
            mask = np.expand_dims(mask, axis=-1)
            # Apply the mask
            # Convert to float32 for the blending operation, and rescale to [0, 1]
            img_seq_float = img_seq[4].astype(np.float32) / 255.0
            img_sub_float = img_sub.astype(np.float32) / 255.0
            # Perform the blending operation
            blended = img_seq_float * (1 - mask) + img_sub_float * mask
            # Convert back to uint8 and rescale to [0, 255] for PIL image
            blended_uint8 = (blended * 255).astype(np.uint8)

            img_seq[4] = blended_uint8
            
            # Set the label to 1
            change = 1

        # Apply irrelevant changes to t5

        # Convert t5 to PIL image
        img_seq[4] = Image.fromarray(img_seq[4])

        # if random.random() < self.config['irrelevant_changes']['seasonal_change_prob']:
        #     img_seq[4] = seasonal_change_transform(img_seq[4])

        # 2023/10/23: We replace default ColorJitter with CustomColorJitter
        img_seq[4] = seasonal_change_transform(img_seq[4])

        # 2023/10/23: remove cloud cover transform for t5, since it would make the problem much harder
        # if random.random() < self.config['irrelevant_changes']['cloud_cover_prob']:
        #     img_seq[4] = cloud_cover_transform(img_seq[4])

        # Convert t5 to numpy array
        img_seq[4] = np.array(img_seq[4])

        """ Get the same augmentations for the image time series (only for training set) """
        if self.mode == 'train':
            # # Augmentations
            # # 1. Random flip
            # #    Left-right flip
            # if random.random() > 0.5:
            #     for i in range(len(img_seq)):
            #         img_seq[i] = img_seq[i][::-1, ...]
            #     if change == 1:
            #         mask = mask[::-1, ...]

            # #    Up-down flip
            # if random.random() > 0.8:
            #     for i in range(len(img_seq)):
            #         img_seq[i] = img_seq[i][:, ::-1, ...]
            #     if change == 1:
            #         mask = mask[:, ::-1, ...]

            # # 2. Random rotation (fixed angles: 90, 180, 270)
            # if random.random() > 0.05:
            #     rot = random.randrange(4)
            #     if rot > 0:
            #         for i in range(len(img_seq)):
            #             img_seq[i] = np.rot90(img_seq[i], k=rot)
            #         if change == 1:
            #             mask = np.rot90(mask, k=rot)

            # 3. Guassian blur + 4. Brightness, contrast, saturation
            for i in range(len(img_seq)):
                if random.random() > 0.98:
                    if random.random() > 0.985:
                        img_seq[i] = gauss_noise(img_seq[i])
                    elif random.random() > 0.985:
                        img_seq[i] = cv2.blur(img_seq[i], (3, 3))
                elif random.random() > 0.98:
                    if random.random() > 0.985:
                        img_seq[i] = saturation(img_seq[i], 0.9+random.random()*0.2)
                    elif random.random() > 0.985:
                        img_seq[i] = brightness(img_seq[i], 0.9+random.random()*0.2)
                    elif random.random() > 0.985:
                        img_seq[i] = contrast(img_seq[i], 0.9+random.random()*0.2)

        """ Z-score normalization """
        for i in range(len(img_seq)):
            img_seq[i] = self.zscore_norm(img_seq[i], self.config['datasets']['norm']['mean'], self.config['datasets']['norm']['std'])

        base_img = self.zscore_norm(base_img, self.config['datasets']['norm']['mean'], self.config['datasets']['norm']['std'])

        """ Convert the images to tensors """
        for i in range(len(img_seq)):
            img_seq[i] = torch.from_numpy(img_seq[i].transpose((2, 0, 1)).copy()).float()
        
        base_img = torch.from_numpy(base_img.transpose((2, 0, 1)).copy()).float()

        if change == 1:
            change_mask = torch.from_numpy(mask.transpose((2, 0, 1)).copy()).float()
        else:
            change_mask = torch.zeros((1, base_img.shape[1], base_img.shape[2])).float()

        change = torch.tensor(change).int()

        """ Return the sample """
        sample = {
            'base': base_img,
            't1': img_seq[0],
            't2': img_seq[1],
            't3': img_seq[2],
            't4': img_seq[3],
            't5': img_seq[4],
            'change': change,
            'mask': change_mask
        }

        return sample


    def add_cloud(self, image):
        """ Add a cloud to the image """
        # image: PIL image

        # Choose random intensity and size for the cloud
        # cloud_intensity = random.uniform(0.7, 1)

        # 2023/10/23:keep transparency to be randomly sampled from [0, 1]
        # cloud_intensity = random.uniform(0.5, 0.7)
        cloud_intensity = random.uniform(0, 1)

        cloud_size = random.randint(image.size[0] // 2, image.size[0]) # Change the size to be up to the image width

        # Create a white cloud with random position
        cloud = Image.new('RGB', image.size, (255, 255, 255))
        cloud_mask = Image.new('L', image.size)
        cloud_mask_draw = ImageDraw.Draw(cloud_mask)

        # Draw a single large cloud
        pos_and_size = [random.randint(0, image.size[0] - cloud_size), random.randint(0, image.size[1] - cloud_size), cloud_size, cloud_size]
        cloud_mask_draw.ellipse(pos_and_size, fill='white')

        # # Blend the image with the cloud
        # image_with_cloud = Image.blend(image, cloud, alpha=ImageEnhance.Brightness(cloud_mask).enhance(cloud_intensity))

        # Enhance the brightness of the cloud mask
        cloud_mask = ImageEnhance.Brightness(cloud_mask).enhance(cloud_intensity)

        # Composite the image with the cloud
        # image_with_cloud = Image.composite(image, cloud, cloud_mask)
        image_with_cloud = Image.composite(cloud, image, cloud_mask)

        return image_with_cloud


    def generate_soft_mask(self, size, apply_gaussian=True):
        """ Generate a random soft mask """
        H = size[0]
        W = size[1]

        center_x = np.random.randint(W)
        center_y = np.random.randint(H)
        # width = np.random.randint(W // 2)
        # height = np.random.randint(H // 2)
        width = np.random.randint(W // 4, W // 2)
        height = np.random.randint(H // 4, H // 2)

        mask = np.zeros((H, W))
        x1 = np.clip(center_x - width // 2, 0, W)
        x2 = np.clip(center_x + width // 2, 0, W)
        y1 = np.clip(center_y - height // 2, 0, H)
        y2 = np.clip(center_y + height // 2, 0, H)
        mask[y1:y2, x1:x2] = 1

        # smooth the mask with a Gaussian filter
        
        # 2023/10/23: set Gaussian filter as a hyper-parameter, in which we can turn on or off
        if apply_gaussian:
            mask = gaussian_filter(mask, sigma=min(width, height) / 3)  

        return mask


    def zscore_norm(self, img, mean, std):
        """ Z-score normalization """
        # Input: numpy array [H, W, C]
        # mean: [C]
        # std: [C]
        # Output: numpy array [H, W, C]
        mean_expanded = np.expand_dims(mean, axis=(0, 1))
        std_expanded = np.expand_dims(std, axis=(0, 1))

        return (img - mean_expanded) / std_expanded


class CustomColorJitter(torch.nn.Module):
    """ 
    Custom ColorJitter transform
    Reference: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#ColorJitter
    
    What is the difference between this and the default ColorJitter?
    for each hyper-parameter, we have 20% possibility to keep the brightness / saturation / contrast / hue as the same
    in the default ColorJitter, we have 20% possibility to keep the brightness + saturation + contrast + hue as the same
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.8):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def forward(self, img):
        # TODO: consider fn_idx = torch.randperm(4) to randomize the order of the transforms

        # adjust brightness
        if random.random() < self.prob:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        # adjust contrast
        if random.random() < self.prob:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = transforms.functional.adjust_contrast(img, contrast_factor)

        # adjust saturation
        if random.random() < self.prob:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            img = transforms.functional.adjust_saturation(img, saturation_factor)

        # adjust hue
        if random.random() < self.prob:
            hue_factor = random.uniform(-self.hue, self.hue)
            img = transforms.functional.adjust_hue(img, hue_factor)

        return img