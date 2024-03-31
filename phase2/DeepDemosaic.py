from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import functional as TF
from torch.nn.functional import interpolate
import time
import gc
import cv2
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from model import Model, createUDnCNN
from dataset_prep import Create_Dataset, CFA_pattern


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def extract_patches(bayer_img, dim, stride):
    patches = []
    locations = []
    h, w = bayer_img.shape[:2]

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            if i + dim > h:
                i = h - dim
            if j + dim > w:
                j = w - dim

            patch = bayer_img[i:i + dim, j:j + dim]
            patches.append(patch)
            locations.append((i, j))

            if j + dim == w:
                break

        if i + dim == h:
            break

    return patches, locations


def inference(bayer_img, trained_model, dim, stride):
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    bayer_img = bayer_img / 255.0
    patches, locations = extract_patches(bayer_img, dim, stride)

    # Output image placeholder
    output_img = np.zeros((3, bayer_img.shape[0], bayer_img.shape[1]), dtype=np.float32)
    count_map = np.zeros((bayer_img.shape[0], bayer_img.shape[1]), dtype=np.float32)
    
    with torch.no_grad():
        for patch, location in zip(patches, locations):
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred = trained_model(patch_tensor)
            pred = pred.squeeze().cpu().numpy()
            
            i, j = location
            output_img[:, i:i + dim, j:j + dim] += pred
            count_map[i:i + dim, j:j + dim] += 1

    # Averaging overlapping areas
    output_img /= count_map[None, :, :]

    output_img = output_img * 255
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img


def calculate_rmse(gt_img, pred_img):
    gt_img = gt_img.astype(np.float64)
    pred_img = pred_img.astype(np.float64)
    mse = mean_squared_error(gt_img.ravel(), pred_img.ravel())
    rmse = sqrt(mse)
    return rmse
