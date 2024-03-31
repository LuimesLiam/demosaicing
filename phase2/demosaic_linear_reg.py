from typing import Tuple
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
import math
from scipy.signal import convolve2d
import torch
import torch.optim as optim
from scipy.ndimage import convolve1d



def clip_histogram(hist, clip_limit):
    # Clip the histogram
    clipped_hist = np.minimum(hist, clip_limit)
    excess_pixels = np.sum(hist - clipped_hist)
    
    # Calculate the number of excess pixels to redistribute evenly
    excess_per_bin = excess_pixels // len(hist)
    
    # Distribute the excess pixels evenly
    clipped_hist += excess_per_bin
    
    # Handle any remaining pixels that couldn't be evenly distributed
    remaining_excess = int(excess_pixels - excess_per_bin * len(hist))
    if remaining_excess > 0:
        step = max(len(hist) // remaining_excess, 1)
        for i in range(0, len(hist), step):
            clipped_hist[i] += 1
            remaining_excess -= 1
            if remaining_excess == 0:
                break
    
    return clipped_hist

def compute_cdf(hist):
    cdf = hist.cumsum()
    return cdf / cdf[-1]

def clahe_custom(image, clip_limit=2.0, grid_size=(8, 8), overlap_ratio=5):
    image =cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    tile_height = l_channel.shape[0] // grid_size[0]
    tile_width = l_channel.shape[1] // grid_size[1]
    
    overlap_height = int(tile_height * overlap_ratio)
    overlap_width = int(tile_width * overlap_ratio)
    
    l_channel_clahe = np.zeros_like(l_channel)
    
    # Iterate over each tile with overlap
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Define the tile region
            y_start = i * tile_height - min(i, 1) * overlap_height
            y_end = (i + 1) * tile_height + min(grid_size[0] - i - 1, 1) * overlap_height
            x_start = j * tile_width - min(j, 1) * overlap_width
            x_end = (j + 1) * tile_width + min(grid_size[1] - j - 1, 1) * overlap_width

            tile = l_channel[y_start:y_end, x_start:x_end]

            hist, _ = np.histogram(tile.flatten(), bins=256, range=[0, 256])
 
            hist_clipped = clip_histogram(hist, clip_limit)

            cdf = compute_cdf(hist_clipped)
            tile_equalized = np.interp(tile.flatten(), np.arange(256), 255 * cdf).reshape(tile.shape)
            l_channel_clahe[y_start:y_end, x_start:x_end] = tile_equalized

    clahe_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))
    clahe_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)
    
    return clahe_image


def clahe2(image, clip_limit=2.0, grid_size=(8, 8)):
    # This function is to see how my custom function does against the built in CLAHE
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    #l_channel_equalized = equalize_hist_custom(l_channel)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    l_channel_clahe = clahe.apply(l_channel)
    
    clahe_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    clahe_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)
    
    return clahe_image


def gray_world_white_balance(image):
    img_float = image.astype(np.float32)
    
    avg_r = np.mean(img_float[:,:,0])
    avg_g = np.mean(img_float[:,:,1])
    avg_b = np.mean(img_float[:,:,2])
    
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b
    img_balanced = img_float * [scale_r, scale_g, scale_b]
    img_balanced = np.clip(img_balanced, 0, 255)
    img_balanced = img_balanced.astype(np.uint8)
    
    return img_balanced


def create_bayer_masks(shape, pattern):
    R_m = np.zeros(shape)
    G_m = np.zeros(shape)
    B_m = np.zeros(shape)

    if pattern == "RGGB":
        R_m[0::2, 0::2] = 1
        G_m[0::2, 1::2] = 1
        G_m[1::2, 0::2] = 1
        B_m[1::2, 1::2] = 1
    elif pattern == "BGGR":
        B_m[0::2, 0::2] = 1
        G_m[0::2, 1::2] = 1
        G_m[1::2, 0::2] = 1
        R_m[1::2, 1::2] = 1
    elif pattern == "GRBG":
        G_m[0::2, 0::2] = 1
        R_m[0::2, 1::2] = 1
        B_m[1::2, 0::2] = 1
        G_m[1::2, 1::2] = 1
    elif pattern == "GBRG":
        G_m[0::2, 0::2] = 1
        B_m[0::2, 1::2] = 1
        R_m[1::2, 0::2] = 1
        G_m[1::2, 1::2] = 1
    else:
        raise ValueError("Unsupported Bayer pattern")
    return R_m, G_m, B_m

def mean_squared_error(imageA, imageB):
    # Ensure the images are in floating point in case they are in uint8
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)

    # Compute the mean squared error between the two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def demosaic(CFA, RK, GK, smoothing= False, norm=True, display=False, balanced=False, he=False, pattern ='RGGB' ):
    #CFA = plt.imread(f'images/mosaiced_noise/{names[ind]}')
    R_m, G_m, B_m = create_bayer_masks(CFA.shape, pattern)
    R = CFA * R_m
    G = CFA * G_m 
    B = CFA * B_m

    # matrix_minR = RK.min()
    # matrix_maxR = RK.max()

    # RK = (RK - matrix_minR) / (matrix_maxR - matrix_minR)

    # matrix_minG = GK.min()
    # matrix_maxG = GK.max()

    # GK = (GK - matrix_minG) / (matrix_maxG - matrix_minG)
    print(R.shape)
    red = convolve2d(R,RK,'same')
    green = convolve2d(G, GK,'same')
    blue =convolve2d(B,  RK,'same')

    if (smoothing):
        smooth =  np.array([
        [1,  2,  1],
        [2, 12,  2],
        [1,  2,  1]]
        )/24

        red = convolve2d(red,smooth,'same')

        green= convolve2d(green,smooth,'same')

        blue = convolve2d(blue,smooth,'same')

    eg = np.dstack([red, green, blue])
    if (norm):
        eg= cv2.normalize(eg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if (he):
        eg =cv2.normalize(src=eg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        eg = clahe_custom(eg)
        
    if(balanced):
        eg =cv2.normalize(src=eg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        eg = gray_world_white_balance(eg)
    if (display):
        plt.imshow(eg)
        plt.title('Mosaic Image')
        plt.show()
    return eg

# patternB = ['Bbest-rggb','dirtyred','red']
# patternA = ['Abest-rggb', 'dirtygreen','green']
# RK = np.load(f"outputs/matrix/{patternB[-1]}.npy")

# GK = np.load(f"outputs/matrix/{patternA[-1]}.npy")

# print("G", GK)
# print("R", RK)
# name = ['bird.png', 'truck.png','snow-dog.png','bird-white.png','close-up-of-tulips-blooming-in-field-royalty-free-image-1584131603.png','Colourfull_of_birds.png','20231230_150122.png']
# #gbrg
# img = demosaic(name,-1,RK,GK, pattern='RGGB')

# plt.imshow(img)
# plt.title('Mosaic Image')
# plt.show()
