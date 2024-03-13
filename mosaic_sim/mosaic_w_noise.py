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

def add_gaussian_noise(image, mean=0, std=10):
    # Ensuring the image is in float32 for accurate noise addition, then converting back
    image_float = np.float32(image)
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noised_image_float = image_float + gauss
    # Clipping values to maintain uint8 range, then converting back to uint8
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
        mosaiced_image = add_gaussian_noise(mosaiced_image)
 
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, mosaiced_image)


