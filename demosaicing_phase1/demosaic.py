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
    clipped_hist = np.minimum(hist, clip_limit)
    excess = hist - clip_limit
    clipped_hist[:-1] += excess[1:]
    clipped_hist[1:] += excess[:-1]
    return clipped_hist

def compute_cdf(hist):
    cdf = hist.cumsum()
    return cdf / cdf[-1]

def clahe_custom(image, clip_limit=2.0, grid_size=(8, 8), overlap_ratio=5):
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
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its components
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply histogram equalization to the L channel
    #l_channel_equalized = equalize_hist_custom(l_channel)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    # Apply CLAHE to the L channel
    l_channel_clahe = clahe.apply(l_channel)
    
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    clahe_lab_image = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    # Convert the CLAHE-enhanced LAB image back to BGR color space
    clahe_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)
    
    return clahe_image


def gray_world_white_balance(image):
    # Convert the image to float32 to avoid overflow during calculations
    img_float = image.astype(np.float32)
    
    # Calculate the average color of the image across each color channel
    avg_r = np.mean(img_float[:,:,0])
    avg_g = np.mean(img_float[:,:,1])
    avg_b = np.mean(img_float[:,:,2])
    
    # Calculate the average gray value
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    # Calculate the scale factors for each color channel
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b
    
    # Apply the scale factors to balance the colors
    img_balanced = img_float * [scale_r, scale_g, scale_b]
    
    # Clip the values to ensure they are within the valid range [0, 255]
    img_balanced = np.clip(img_balanced, 0, 255)
    
    # Convert the image back to uint8 format
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
    # Add similar conditions for other patterns ("BGGR", "GRBG", "GBRG")
    return R_m, G_m, B_m

def mean_squared_error(imageA, imageB):
    # Ensure the images are in floating point in case they are in uint8
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)

    # Compute the mean squared error between the two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def demosaic(names, ind, RK, GK, smoothing= False, norm=False, display=True, is_gt=False, balanced=False, he=False, pattern ='RGGB' ):
    CFA = plt.imread(f'images/mosaiced_noise/{names[ind]}')
    if is_gt == True:
        gt = plt.imread(f'images/ground_truth/{names[ind]}')

    R_m, G_m, B_m = create_bayer_masks(CFA.shape, pattern)
    R = CFA * R_m
    G = CFA * G_m 
    B = CFA * B_m
    
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

    if (is_gt == True ):
        mse = mean_squared_error(eg, gt)
        print(f"Mean Squared Error: {mse}")
    if (display):
        plt.imshow(eg)
        plt.title('Mosaic Image')
        plt.show()
        

patternB = ['Bbest-rggb','dirtyred']
patternA = ['Abest-rggb', 'dirtygreen']
RK = np.load(f"outputs/matrix/{patternB[0]}.npy")

GK = np.load(f"outputs/matrix/{patternA[0]}.npy")

print("G", GK)
print("R", RK)
name = ['bird.png', 'truck.png','snow-dog.png','bird-white.png','kodim05.png']
#gbrg
demosaic(name,1,RK,GK,smoothing=True, is_gt=False,  he=False, balanced=False, pattern='RGGB')


