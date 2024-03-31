
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
from typing import Tuple
import cv2
import rawpy
import imageio
from PIL import Image


def bggr_mosaic(rgb_image):
    rows, columns, _ = rgb_image.shape
    mosaiced_image = np.zeros((rows, columns), dtype=np.uint8)

    for col in range(columns):
        for row in range(rows):
            if col % 2 == 0 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 0]  # Red
            elif col % 2 == 0 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 1 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 1 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 2]  # Blue

    return mosaiced_image

def grbg_mosaic(rgb_image):
    rows, columns, _ = rgb_image.shape
    mosaiced_image = np.zeros((rows, columns), dtype=np.uint8)

    for col in range(columns):
        for row in range(rows):
            if col % 2 == 0 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 0 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 0]  # Blue
            elif col % 2 == 1 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 2]  # Red
            elif col % 2 == 1 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green

    return mosaiced_image

def rggb_mosaic(rgb_image):
    print(rgb_image.shape)
    rows, columns, _ = rgb_image.shape
    mosaiced_image = np.zeros((rows, columns), dtype=np.uint8)

    for col in range(columns):
        for row in range(rows):
            if col % 2 == 0 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 2]  # Red
            elif col % 2 == 0 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 1 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 1 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 0]  # Blue

    return mosaiced_image

def gbrg_mosaic(rgb_image):
    rows, columns, _ = rgb_image.shape
    mosaiced_image = np.zeros((rows, columns), dtype=np.uint8)

    for col in range(columns):
        for row in range(rows):
            if col % 2 == 0 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green
            elif col % 2 == 0 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 2]  # Red
            elif col % 2 == 1 and row % 2 == 0:
                mosaiced_image[row, col] = rgb_image[row, col, 0]  # Blue
            elif col % 2 == 1 and row % 2 == 1:
                mosaiced_image[row, col] = rgb_image[row, col, 1]  # Green

    return mosaiced_image

def CFA_pattern(img, pattern):
    print(pattern)
    if(pattern == "rggb"):
        return rggb_mosaic(img)
    elif (pattern == 'bggr'):
        return bggr_mosaic(img)
    elif (pattern == 'gbrg'):
        return gbrg_mosaic(img)
    elif (pattern =='grbg'):
        return grbg_mosaic(img)

def extract_image_patches( original_image, bayer_image, patch_size, num_samples):
    img_height, img_width, _ = original_image.shape
    patches_orig = []
    patches_bayer = []

    stride_h = 2
    stride_w = 2

    num_patches_height = (img_height - patch_size) // stride_h + 1
    num_patches_width = (img_width - patch_size) // stride_w + 1
    num_patches = num_patches_height * num_patches_width

    while num_patches > num_samples:
        stride_h += 2
        stride_w += 2
        num_patches_height = (img_height - patch_size) // stride_h + 1
        num_patches_width = (img_width - patch_size) // stride_w + 1
        num_patches = num_patches_height * num_patches_width

    start_y = 0 if img_height % 2 == 0 else 1
    start_x = 0 if img_width % 2 == 0 else 1

    for y in range(start_y, img_height - patch_size + 1, stride_h):
        for x in range(start_x, img_width - patch_size + 1, stride_w):
            patch_orig = original_image[y:y + patch_size, x:x + patch_size]
            patch_bayer = bayer_image[y:y + patch_size, x:x + patch_size]

            patches_orig.append(patch_orig)
            patches_bayer.append(patch_bayer)

    all_possible_patches = [(y, x) for y in range(start_y, img_height - patch_size + 1, 2)
                            for x in range(start_x, img_width - patch_size + 1, 2)]

    remaining_patches = list(set(all_possible_patches) - set(zip(range(start_y, img_height - patch_size + 1, stride_h), 
                                                                range(start_x, img_width - patch_size + 1, stride_w))))

    if len(patches_orig) < num_samples:
        needed_patches = num_samples - len(patches_orig)
        random_indices = random.sample(remaining_patches, needed_patches)
        for (y, x) in random_indices:
            patch_orig = original_image[y:y + patch_size, x:x + patch_size]
            patch_bayer = bayer_image[y:y + patch_size, x:x + patch_size]
            patches_orig.append(patch_orig)
            patches_bayer.append(patch_bayer)

    return patches_orig, patches_bayer

class Create_Dataset:

    def __init__(self, path, ground_truth_path="ground_truth", CFA_path ="mosaiced_noisetest", pattern='bggr'):
        self.path = path
        self.ground_truth_path = ground_truth_path
        self.CFA_path = CFA_path
        self.pattern = pattern
        print(self.pattern)

    def prep_data(self, dim, sample_num):
        CFA_images = []
        ground_truths =[]
        for file_name in sorted(os.listdir(f'{self.path}{self.CFA_path}')):
            if True:
                try:
                    gt = cv2.imread(f'{self.path}{self.ground_truth_path}/{file_name}',cv2.IMREAD_COLOR)
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    gt = np.array(gt)

                    CFA = cv2.imread(f'{self.path}{self.CFA_path}/{file_name}')
                    CFA =  CFA_pattern(CFA, self.pattern).astype(np.float32)
                    CFA = np.expand_dims(CFA, axis=-1)
                except:
                    print("RAW")
                    with rawpy.imread(f'{self.path}{self.ground_truth_path}/{file_name}') as raw:
                        # Postprocess the raw data minimally
                        gt = raw.postprocess()
                        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                        gt = np.array(gt)

                    #img_ref = read_arw('images/sony/long/00001_00_10s.ARW')

                    with rawpy.imread(f'{self.path}{self.CFA_path}/{file_name}') as raw:
                        # Postprocess the raw data minimally
                        CFA = raw.postprocess(no_auto_bright=True)
                        #CFA = cv2.cvtColor(CFA, cv2.COLOR_BGR2RGB)
                        CFA =  CFA_pattern(CFA, self.pattern).astype(np.float32)
                        CFA = np.expand_dims(CFA, axis=-1)
                
                
                gt_patches, CFA_patches = extract_image_patches(gt, CFA, dim, sample_num)
                
                CFA_images.extend(CFA_patches)
                ground_truths.extend(gt_patches)
        return CFA_images, ground_truths
    
    
    def train_test(self, bayer_tensors, original_tensors, test_ratio=0.2, seed=42):
        # Ensure reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Calculate the size of the dataset and the test set
        total_size = bayer_tensors.size(0)
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size

        # Generate shuffled indices
        indices = torch.randperm(total_size)

        # Split indices for training and test sets
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Split the datasets
        bayer_train = bayer_tensors[train_indices]
        bayer_test = bayer_tensors[test_indices]
        original_train = original_tensors[train_indices]
        original_test = original_tensors[test_indices]

        return bayer_train, bayer_test, original_train, original_test
    
    def make_dataset(self, patch_size=32, sample_num = 50, batch_size=16):
        CFA_imgs , gt_imgs = self.prep_data(patch_size,sample_num)
        CFA_tensor = torch.stack([torch.tensor(img.astype(np.float32), dtype=torch.float32)
                                  .permute(2,0,1) / 255. for img in CFA_imgs])
        gt_tensors = torch.stack([torch.tensor(img.astype(np.float32), dtype=torch.float32)
                                  .permute(2,0,1) / 255. for img in gt_imgs])
        CFA_train, CFA_test, gt_train, gt_test = self.train_test(
            CFA_tensor, gt_tensors, 0.2, 42
        )
        #self.train_test(CFA_tensor, gt_tensors)
        
        train_dataset = TensorDataset(CFA_train, gt_train)
        test_dataset = TensorDataset(CFA_test, gt_test)

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader





