import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the folder containing input images
input_folder = 'images/ground_truth'

# Path to the folder where mosaiced images will be saved
output_folder = 'images/mosaiced_noise'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def add_color_specific_gaussian_noise(mosaiced_image, std_green=1, std_red=2, std_blue=4, mosaic_pattern='rggb'):
    mosaiced_image_float = np.float32(mosaiced_image)
    std_green += random.uniform(-0.1, 0.1)
    std_red += random.uniform(-0.1, 0.1)
    std_blue += random.uniform(-0.1, 0.1)

    noise_green = np.random.normal(0, std_green, mosaiced_image.shape).astype(np.float32)
    noise_red = np.random.normal(0, std_red, mosaiced_image.shape).astype(np.float32)
    noise_blue = np.random.normal(0, std_blue, mosaiced_image.shape).astype(np.float32)

    noised_image = np.zeros_like(mosaiced_image_float)

    rows, cols = mosaiced_image.shape
    for i in range(rows):
        for j in range(cols):
            if mosaic_pattern == 'rggb':
                if i % 2 == 0 and j % 2 == 0:  # Red
                    noised_image[i, j] = mosaiced_image_float[i, j] + noise_red[i, j]
                elif i % 2 == 1 and j % 2 == 1:  # Blue
                    noised_image[i, j] = mosaiced_image_float[i, j] + noise_blue[i, j]
                else:  # Green
                    noised_image[i, j] = mosaiced_image_float[i, j] + noise_green[i, j]
    
    noised_image_uint8 = np.clip(noised_image, 0, 255).astype('uint8')
    
    return noised_image_uint8

def add_gaussian_noise(image, mean=0, std=10):
    image_float = np.float32(image)
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noised_image_float = image_float + gauss
    noised_image_uint8 = np.clip(noised_image_float, 0, 255).astype('uint8')
    return noised_image_uint8


# Path to the folder containing input images
input_folder = 'images/ground_truth'

# Path to the folder where mosaiced images will be saved
output_folder = 'images/mosaiced_noise'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

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

# Choose the mosaic pattern
mosaic_pattern = 'rggb'  # You can change this to 'grbg', 'gbrg', 'bggr'

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.bmp')):  # Assuming images have common extensions
        # Read the image
        file_path = os.path.join(input_folder, filename)
        rgb_image = cv2.imread(file_path)

        # Create mosaiced image based on the chosen pattern
        if mosaic_pattern == 'bggr':
            mosaiced_image = bggr_mosaic(rgb_image)
        elif mosaic_pattern == 'grbg':
            mosaiced_image = grbg_mosaic(rgb_image)
        elif mosaic_pattern == 'rggb':
            mosaiced_image = rggb_mosaic(rgb_image)
        elif mosaic_pattern == 'gbrg':
            mosaiced_image = gbrg_mosaic(rgb_image)
        mosaiced_image = add_color_specific_gaussian_noise(mosaiced_image)
 
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, mosaiced_image)


