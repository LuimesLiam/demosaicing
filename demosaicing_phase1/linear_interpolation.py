
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.exposure
from typing import Tuple
from scipy.signal import convolve2d
from sklearn.metrics import mean_squared_error

def bayer(im):
    r = np.zeros(im.shape[:2])
    g = np.zeros(im.shape[:2])
    b = np.zeros(im.shape[:2])
    r[0::2, 0::2] += im[0::2, 0::2]
    g[0::2, 1::2] += im[0::2, 1::2]
    g[1::2, 0::2] += im[1::2, 0::2]
    b[1::2, 1::2] += im[1::2, 1::2]
    return r, g, b

def demosaic_bilin(CFA: np.ndarray, pattern: str = "RGGB", ground_truth: np.ndarray = None):

    r, g, b = bayer(CFA)
    
    k_g = 1/4 * np.array([
        [0,1,0],
        [1,0,1],
        [0,1,0]])
    convg =convolve2d(g, k_g, 'same')
    g = g + convg

    # red interpolation
    k_r_1 = 1/4 * np.array([
        [1,0,1],
        [0,0,0],
        [1,0,1]])
    convr1 =convolve2d(r, k_r_1, 'same')
    convr2 =convolve2d(r+convr1, k_g, 'same')
    r = r + convr1 + convr2

    # blue interpolation
    k_b_1 = 1/4 * np.array([
        [1,0,1],
        [0,0,0],
        [1,0,1]])
    convb1 =convolve2d(b, k_b_1, 'same')
    convb2 =convolve2d(b+convb1, k_g, 'same')
    b = b + convb1 + convb2
    demosaiced_image = np.stack((r,g,b), axis=2)
    demosaiced_image= cv2.normalize(demosaiced_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return demosaiced_image

def mean_squared_error(imageA, imageB):
    # Ensure the images are in floating point in case they are in uint8
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)

    # Compute the mean squared error between the two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

name = ['bird.png', 'truck.png']
ind = 1
mosaic_images_dir = f'images/mosaiced/{name[ind]}'
ground_truth_images_dir = f'images/ground_truth/{name[ind]}'

ground_tr_img = plt.imread(ground_truth_images_dir)
example_mosaic_image = plt.imread(mosaic_images_dir)
example_demosaiced_image = demosaic_bilin(example_mosaic_image)

print(ground_tr_img.shape, example_demosaiced_image.shape)


plt.imshow(example_demosaiced_image)
plt.axis("off")

mse = mean_squared_error(example_demosaiced_image, ground_tr_img)
print(mse)
plt.show()

