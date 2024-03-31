
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, convolve1d  
from sklearn.linear_model import LinearRegression
from PIL import Image
import cv2
def apply_horizontal_filter(x, y):
    return convolve1d(x, y, mode="mirror")

def apply_vertical_filter(x, y):
    return convolve1d(x, y, mode="mirror", axis=0)

def apply_gaussian_filter(image, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma)



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

def horizontal_convolution(image, filter_kernel):
    return convolve(image, filter_kernel[None, :], mode='mirror')

def vertical_convolution(image, filter_kernel):
    return convolve(image, filter_kernel[:, None], mode='mirror')

def demosaic_CFA_Bayer(color_filter_array, bayer_pattern="RGGB"):
    cfa_data = np.squeeze(np.array(color_filter_array, dtype=float))

    red_mask, green_mask, blue_mask = create_bayer_masks(cfa_data.shape, bayer_pattern)

    filter_basic = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
    filter_correction = np.array([-0.25, 0.0, 0.5, 0.0, -0.25])


    red_channel = cfa_data * red_mask
    green_channel = cfa_data * green_mask
    blue_channel = cfa_data * blue_mask

    green_horiz = np.where(green_mask == 0, horizontal_convolution(cfa_data, filter_basic) + horizontal_convolution(cfa_data, filter_correction), green_channel)
    green_vert = np.where(green_mask == 0, vertical_convolution(cfa_data, filter_basic) + vertical_convolution(cfa_data, filter_correction), green_channel)

    color_diff_horiz = np.where(red_mask == 1, red_channel - green_horiz, np.where(blue_mask == 1, blue_channel - green_horiz, 0))
    color_diff_horiz = np.where(blue_mask == 1, red_channel - green_horiz, np.where(blue_mask == 1, blue_channel - green_horiz, color_diff_horiz))

    color_diff_vert = np.where(red_mask == 1, red_channel - green_vert, np.where(blue_mask == 1, blue_channel - green_vert, 0))
    color_diff_vert = np.where(blue_mask == 1, red_channel - green_vert, np.where(blue_mask == 1, blue_channel - green_vert, color_diff_vert))
    smoothness_horiz = np.abs(color_diff_horiz - np.pad(color_diff_horiz, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    smoothness_vert = np.abs(color_diff_vert - np.pad(color_diff_vert, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    smooth_kernel = np.array([[0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 3, 0, 3], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1]])

    smoothness_horiz_conv = convolve(smoothness_horiz, smooth_kernel, mode="constant")
    smoothness_vert_conv = convolve(smoothness_vert, np.transpose(smooth_kernel), mode="constant")


    interp_direction_mask = smoothness_vert_conv >= smoothness_horiz_conv
    G = np.where(interp_direction_mask, green_horiz, green_vert)
    M = np.where(interp_direction_mask, 1, 0)

    kernel_b = np.array([0.5, 0, 0.5])
    R_row = np.transpose(np.any(red_mask == 1, axis=1)[None]) * np.ones(red_channel.shape)
    B_row = np.transpose(np.any(blue_mask == 1, axis=1)[None]) * np.ones(blue_channel.shape)

    R = np.where(
        (green_mask == 1) & (R_row == 1),
        G + apply_horizontal_filter(red_channel, kernel_b) - apply_horizontal_filter(G, kernel_b),
        red_channel,
    )

    R = np.where(
        (green_mask == 1) & (B_row == 1),
        G + apply_vertical_filter(red_channel, kernel_b) - apply_vertical_filter(G, kernel_b),
        R,
    )

    B = np.where(
        (green_mask == 1) & (B_row == 1),
        G + apply_horizontal_filter(blue_channel, kernel_b) - apply_horizontal_filter(G, kernel_b),
        blue_channel,
    )

    B = np.where(
        (green_mask == 1) & (R_row == 1),
        G + apply_vertical_filter(blue_channel, kernel_b) - apply_vertical_filter(G, kernel_b),
        B,
    )

    R = np.where(
        (B_row == 1) & (blue_mask == 1),
        np.where(
            M == 1,
            B + apply_horizontal_filter(R, kernel_b) - apply_horizontal_filter(B, kernel_b),
            B + apply_vertical_filter(R, kernel_b) - apply_vertical_filter(B, kernel_b),
        ),
        R,
    )

    B = np.where(
        (R_row == 1) & (red_mask == 1),
        np.where(
            M == 1,
            R + apply_horizontal_filter(B, kernel_b) - apply_horizontal_filter(R, kernel_b),
            R + apply_vertical_filter(B, kernel_b) - apply_vertical_filter(R, kernel_b),
        ),
        B,
    )
    final_image = np.stack([R,G,B], axis=-1)


    return final_image


def mean_squared_error(imageA, imageB):
    # Ensure the images are in floating point in case they are in uint8
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)

    # Compute the mean squared error between the two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

print("start")
names = ['bird.png', 'truck.png','snow-dog.png','bird-white.png']
ind = 1

CFA = plt.imread(f'images/mosaiced_noise/{names[ind]}')
gt = plt.imread(f'images/ground_truth/{names[ind]}')
eg=demosaic_CFA_Bayer(CFA, 'RGGB')
#eg= cv2.normalize(eg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


mse = mean_squared_error(eg, gt)
print(f"Mean Squared Error: {mse}")

plt.imshow(eg)
#plt.title('Mosaic Image')
plt.axis('off')
plt.show()
print("end")
