from matplotlib import pyplot as plt
import cv2
from DeepDemosaic import inference, extract_patches, calculate_rmse
from model import Model, createUDnCNN
from dataset_prep import Create_Dataset, CFA_pattern
import os 
import torch
import numpy as np
from demosaic_linear_reg import demosaic,gray_world_white_balance
from other import clahe_custom
import rawpy
import imageio


def add_color_shift_to_raw_image(rgb, shift_values):


    # Apply the color shift
    # Note: The image data should be in 16-bit per channel format for better color depth.
    # Convert to float for manipulation, then back to uint16 if needed.
    rgb_shifted = rgb.astype(np.float32)
    rgb_shifted[:, :, 0] += shift_values[0]  # R channel
    rgb_shifted[:, :, 1] += shift_values[1]  # G channel
    rgb_shifted[:, :, 2] += shift_values[2]  # B channel

    # Ensure the shifted values remain within the valid range
    rgb_shifted = np.clip(rgb_shifted, 0, 255).astype(np.uint8)
    return rgb_shifted
    # Save the processed image
    #imageio.imwrite(output_file_path, rgb_shifted)



def add_color_cast(img, cast_type='blue', intensity=25):
    if cast_type == 'blue':
        img[:, :, 0] = cv2.add(img[:, :, 0], intensity)
    elif cast_type == 'red':
        img[:, :, 2] = cv2.add(img[:, :, 2], intensity)
    elif cast_type == 'green':
        img[:, :, 1] = cv2.add(img[:, :, 1], intensity)
    return img


def deep(CFA, net_name='model_state_dict.pth'):
    save_dir = ''
    trained_net, lossfun, optimizer= createUDnCNN()
    model_path = os.path.join(save_dir, net_name)
    trained_net.load_state_dict(torch.load(model_path))
    bayer_img = CFA_pattern(CFA,'grbg')
    


    print(bayer_img.shape)
    stride = 10
    dim = 16
    prediction = inference(bayer_img, trained_net, dim, stride)
    pred_img = np.transpose(prediction, (1, 2, 0))
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    return pred_img




def lin(CFA):
    patternB = ['Bbest-rggb','dirtyred','red_times20', 'red_V2']
    patternA = ['Abest-rggb', 'dirtygreen','green_times20', 'green_V2']
    RK = np.load(f"outputs/matrix/{patternB[-1]}.npy")

    GK = np.load(f"outputs/matrix/{patternA[-1]}.npy")

    print("G", GK)
    print("R", RK)
    #gbrg
    #CFA = plt.imread(f'images/mosaiced_noise/{name[ind]}')
    img = demosaic(CFA,RK,GK,norm=False, pattern='RGGB')
    return img


name = [ 'truck.png']

CFA = plt.imread(f'images/mosaiced_noise/{name[-1]}')
CFA_DEEP = cv2.imread(f'images/mosaiced_noise/{name[-1]}')


# img = lin(CFA)

# img = deep(CFA)


######### simulated data: 



###################### Histogram ################
# img_shift = np.float32(img) / 255.0 *0.3
# img_hist = clahe_custom(img_shift)


# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax[0].imshow(img_shift)
# ax[0].set_title('Dark')
# ax[0].axis('off')  

# ax[1].imshow(img_hist)
# ax[1].set_title('Histogram')
# ax[1].axis('off')  
# plt.show()

# ###################### White Balance #################
# hue_shift_value = 1
# img =cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# img_converted = img.astype(np.float32)

# shifted_image = add_color_cast(img)

# img_gray = gray_world_white_balance(shifted_image)


# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax[0].imshow(shifted_image)
# ax[0].set_title('Hue shifted')
# ax[0].axis('off')  

# ax[1].imshow(img_gray)
# ax[1].set_title('White Blanaced')
# ax[1].axis('off')  
# plt.show()



############### short



with rawpy.imread('images/sony/test2.ARW') as raw:
    # Postprocess the raw data minimally
    img = raw.postprocess(no_auto_bright=True)
    #img = CFA_pattern(img1,'grbg').astype(np.float32)


#img_ref = read_arw('images/sony/long/00001_00_10s.ARW')

# with rawpy.imread('images/sony/long/bikes.ARW') as raw:
#     # Postprocess the raw data minimally
#     img_ref = raw.postprocess()

img_pred = deep (img, 'model_state_dict_raw_raw.pth')
#deep(img,'model_state_dict.pth')


#img_pred_lowLight= deep (img, 'model_state_dict_raw_raw.pth')


fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax[0].imshow(img)
ax[0].set_title('Dark')
ax[0].axis('off')  



ax[1].imshow(img_pred)
ax[1].set_title('Normal')
ax[1].axis('off')  

# ax[2].imshow(img_pred_lowLight)
# ax[2].set_title("Low Light Model")
# ax[2].axis('off')
plt.show()

# img = img_pred
# img_hist = clahe_custom(img)


# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax[0].imshow(img)
# ax[0].set_title('Dark')
# ax[0].axis('off')  

# ax[1].imshow(img_hist)
# ax[1].set_title('Histogram')
# ax[1].axis('off')  
# plt.show()

img_hist = clahe_custom(img_pred)

# img_hist_low_light_model = clahe_custom(img_pred_lowLight, clip_limit=2.0)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax[0].imshow(img)
ax[0].set_title('Dark')
ax[0].axis('off')  

ax[1].imshow(img_hist)
ax[1].set_title('Histogram')
ax[1].axis('off')  

# ax[2].imshow(img_hist_low_light_model)
# ax[2].set_title('Histogram With Low Light Model')
# ax[2].axis('off')  
plt.show()


img_col = add_color_shift_to_raw_image(img_pred, (100,10,15))


fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

img_gray = gray_world_white_balance(img_col)

ax[0].imshow(img_col)
ax[0].set_title('Colour shifted')
ax[0].axis('off')  

ax[1].imshow(img_gray)
ax[1].set_title('White Balanced')
ax[1].axis('off')  
plt.show()

### colour shift






