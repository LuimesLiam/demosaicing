import numpy as np
from scipy.ndimage import convolve, convolve1d
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, sobel

def apply_anti_aliasing_filter(image, sigma=1.5):
    """
    Apply a Gaussian filter for anti-aliasing. The sigma controls the strength
    of smoothing. A larger sigma value will result in more smoothing, which can
    help in reducing aliasing but may also blur the image.
    """
    return gaussian_filter(image, sigma=sigma)

def edge_aware_smoothing(image):
    """
    An example function that demonstrates edge-aware smoothing. This function
    uses the Sobel operator to detect edges and applies a mild Gaussian
    smoothing. The smoothing is less aggressive near edges to preserve detail.
    """
    edge_magnitude = np.sqrt(sobel(image, axis=0)**2 + sobel(image, axis=1)**2)
    edge_magnitude = edge_magnitude / edge_magnitude.max()  # Normalize
    
    # Smooth the image
    smoothed_image = gaussian_filter(image, sigma=1)
    
    # Blend the original and smoothed images based on edge magnitude
    # This is a simplified approach; more sophisticated methods could be used
    k = 0.5  # Blend factor; could be adapted based on edge strength
    final_image = k * image + (1 - k) * smoothed_image * (1 - edge_magnitude)
    
    return final_image

def simulate_rggb_cfa(image_shape):
    # Creates a simulated RGGB CFA pattern for demonstration
    rggb_image = np.zeros(image_shape)
    # Simulate RGGB pattern
    rggb_image[::2, ::2] = 1 # Red
    rggb_image[::2, 1::2] = 2 # Green on red row
    rggb_image[1::2, ::2] = 2 # Green on blue row
    rggb_image[1::2, 1::2] = 3 # Blue
    return rggb_image

def calculate_gradients(rggb_image):
    # Calculate horizontal and vertical gradients for green interpolation
    # Use simple forward differences for gradient calculation
    vertical_gradient = np.abs(np.diff(rggb_image, axis=0, append=rggb_image[-1:]))
    horizontal_gradient = np.abs(np.diff(rggb_image, axis=1, append=rggb_image[:,-1:]))

    return horizontal_gradient, vertical_gradient

def directional_interpolate_green(rggb_image, GK):
    height, width = rggb_image.shape
    green_channel = np.zeros_like(rggb_image)

    # Pre-fill known green values
    green_channel[::2, 1::2] = rggb_image[::2, 1::2] # Green on red rows
    green_channel[1::2, ::2] = rggb_image[1::2, ::2] # Green on blue rows

    # Calculate gradients
    h_grad, v_grad = calculate_gradients(rggb_image)

    # Interpolate green based on gradient direction
    for i in range(height):
        for j in range(width):
            # Skip if green is already known
            if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                continue

            if h_grad[i, j] < v_grad[i, j]:
                # Horizontal gradient is smaller, interpolate horizontally
                green_channel[i, j] = np.mean(rggb_image[i, max(j-1, 0):j+2:2])
            else:
                # Vertical gradient is smaller, interpolate vertically
                green_channel[i, j] = np.mean(rggb_image[max(i-1, 0):i+2:2, j])

    # Convolve with GK for smoothing, applied to all green pixels for simplicity
    green_channel = convolve(green_channel, GK)

    return green_channel

def interpolate_red_blue(rggb_image, GK, RBK):
    height, width = rggb_image.shape
    # Initialize channels
    red_channel = np.zeros_like(rggb_image)
    blue_channel = np.zeros_like(rggb_image)
    
    # Directly copy known red and blue values from the RGGB pattern
    red_channel[::2, ::2] = rggb_image[::2, ::2]  # Red pixels
    blue_channel[1::2, 1::2] = rggb_image[1::2, 1::2]  # Blue pixels

    # Calculate gradients for red and blue interpolation
    h_grad, v_grad = calculate_gradients(rggb_image)

    # Interpolate Red
    for i in range(0, height, 2):
        for j in range(1, width, 2):  # Red pixels are missing here in RGGB
            if h_grad[i, j] < v_grad[i, j]:
                # Horizontal interpolation
                red_channel[i, j] = np.mean([rggb_image[i, max(j-2, 0)], rggb_image[i, min(j+2, width-1)]])
            else:
                # Vertical interpolation
                red_channel[i, j] = np.mean([rggb_image[max(i-2, 0), j], rggb_image[min(i+2, height-1), j]])

    # Interpolate Blue
    for i in range(1, height, 2):
        for j in range(0, width, 2):  # Blue pixels are missing here in RGGB
            if h_grad[i, j] < v_grad[i, j]:
                # Horizontal interpolation
                blue_channel[i, j] = np.mean([rggb_image[i, max(j-2, 0)], rggb_image[i, min(j+2, width-1)]])
            else:
                # Vertical interpolation
                blue_channel[i, j] = np.mean([rggb_image[max(i-2, 0), j], rggb_image[min(i+2, height-1), j]])

    # Apply RBK for smoothing
    red_channel = convolve(red_channel, RBK)
    blue_channel = convolve(blue_channel, RBK)

    return red_channel, blue_channel


patternB = ['Bbest-rggb','dirtyred']
patternA = ['Abest-rggb', 'dirtygreen']
RK = np.load(f"outputs/matrix/{patternB[0]}.npy")

GK = np.load(f"outputs/matrix/{patternA[0]}.npy")

print("G", GK)
print("R", RK)
names = ['bird.png', 'truck.png','snow-dog.png','bird-white.png','kodim05.png']
#gbrg
CFA = plt.imread(f'images/mosaiced/{names[1]}')

green_channel = directional_interpolate_green(CFA, GK)
red_channel, blue_channel = interpolate_red_blue(CFA, GK, RK)

img = np.stack((red_channel, green_channel, blue_channel), axis=-1)
img = apply_anti_aliasing_filter(img)

# Apply edge-aware smoothing
img = edge_aware_smoothing(img)

img =cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


plt.imshow(img)
plt.title('Mosaic Image')
plt.show()












# import numpy as np
# from typing import Tuple
# import numpy as np
# from scipy.ndimage import convolve
# import matplotlib.pyplot as plt
# import cv2
# import math
# from scipy.signal import convolve2d
# import torch
# import torch.optim as optim
# from scipy.ndimage import convolve1d

# def create_bayer_masks(shape, pattern):
#     R_m = np.zeros(shape)
#     G_m = np.zeros(shape)
#     B_m = np.zeros(shape)

#     if pattern == "RGGB":
#         R_m[0::2, 0::2] = 1
#         G_m[0::2, 1::2] = 1
#         G_m[1::2, 0::2] = 1
#         B_m[1::2, 1::2] = 1
#     elif pattern == "BGGR":
#         B_m[0::2, 0::2] = 1
#         G_m[0::2, 1::2] = 1
#         G_m[1::2, 0::2] = 1
#         R_m[1::2, 1::2] = 1
#     elif pattern == "GRBG":
#         G_m[0::2, 0::2] = 1
#         R_m[0::2, 1::2] = 1
#         B_m[1::2, 0::2] = 1
#         G_m[1::2, 1::2] = 1
#     elif pattern == "GBRG":
#         G_m[0::2, 0::2] = 1
#         B_m[0::2, 1::2] = 1
#         R_m[1::2, 0::2] = 1
#         G_m[1::2, 1::2] = 1
#     else:
#         raise ValueError("Unsupported Bayer pattern")
#     return R_m, G_m, B_m

# def edge_sensing(kernel):
#     """
#     Detects edges within a 5x5 kernel and returns the direction of interpolation.
    
#     :param kernel: A 5x5 numpy array, the local neighborhood of a pixel.
#     :return: The direction of interpolation: 'H' for horizontal, 'V' for vertical.
#     """
#     # Compute horizontal and vertical differences
#     diff_horizontal = np.abs(np.sum(kernel[2, :2]) - np.sum(kernel[2, 3:]))
#     diff_vertical = np.abs(np.sum(kernel[:2, 2]) - np.sum(kernel[3:, 2]))

#     # Determine the direction with the smallest difference (less likely to be an edge)
#     if diff_horizontal < diff_vertical:
#         return 'H'  # Horizontal direction has less edge
#     else:
#         return 'V'  # Vertical direction has less edge

# import numpy as np

# def directional_interpolation(bayer_image, GK, RBK):
#     height, width = bayer_image.shape
#     demosaiced_image = np.zeros((height, width, 3), dtype=np.uint8)
    
#     R_m, G_m, B_m = create_bayer_masks(bayer_image.shape, 'RGGB')
#     R = CFA * R_m
#     G = CFA * G_m 
#     B = CFA * B_m

#     padded_image = np.pad(bayer_image, 2, mode='reflect')
#     R_pad = np.pad(R, 2, mode='reflect')
#     G_pad = np.pad(G, 2, mode='reflect')
#     B_pad = np.pad(B, 2, mode='reflect')

#     for i in range(height):
#         for j in range(width):
#             kernel = padded_image[i:i+5, j:j+5]
#             kernelR = R_pad[i:i+5, j:j+5]
#             kernelG = G_pad[i:i+5, j:j+5]
#             kernelB = B_pad[i:i+5, j:j+5]
            
#             # Determine color channel based on position in the RGGB pattern
#             if i % 2 == 0:
#                 if j % 2 == 0:  # Red position
#                     color_channel = 'R'
#                 else:  # Green position
#                     color_channel = 'G'
#             else:
#                 if j % 2 == 0:  # Green position
#                     color_channel = 'G'
#                 else:  # Blue position
#                     color_channel = 'B'
            
#             # Apply GK for green everywhere since it's the most common
#             green_comp = np.sum(kernelG * GK)
            
#             # Apply RBK based on the color channel
#             if color_channel in ['R', 'B']:
#                 if color_channel == 'R':
#                     red_or_blue_comp = np.sum(kernelR * RBK)
#                     demosaiced_image[i, j, 0] = red_or_blue_comp  # Red
#                 else:
#                     red_or_blue_comp = np.sum(kernelB * RBK)
#                     demosaiced_image[i, j, 2] = red_or_blue_comp  # Blue
#             else:  # For green position in RGGB, green is directly available, but we already computed it
#                 pass
            
#             demosaiced_image[i, j, 1] = green_comp  # Assign green for all
            

#     demosaiced_image =cv2.normalize(src=demosaiced_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     return demosaiced_image




# patternB = ['Bbest-rggb','dirtyred']
# patternA = ['Abest-rggb', 'dirtygreen']
# RK = np.load(f"outputs/matrix/{patternB[0]}.npy")

# GK = np.load(f"outputs/matrix/{patternA[0]}.npy")

# print("G", GK)
# print("R", RK)
# names = ['bird.png', 'truck.png','snow-dog.png','bird-white.png','kodim05.png']
# #gbrg
# CFA = plt.imread(f'images/mosaiced/{names[1]}')
# img = directional_interpolation(CFA, GK, RK)


# plt.imshow(img)
# plt.title('Mosaic Image')
# plt.show()

