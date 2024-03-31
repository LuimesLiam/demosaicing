from matplotlib import pyplot as plt
import cv2
from DeepDemosaic import inference, extract_patches, calculate_rmse
from model import Model, createUDnCNN
from dataset_prep import Create_Dataset, CFA_pattern
import os 
import torch
import numpy as np
from demosaic_linear_reg import demosaic, clahe_custom,gray_world_white_balance, clahe2

import rawpy
import imageio


def add_color_cast(img, cast_type='blue', intensity=25):
    if cast_type == 'blue':
        img[:, :, 0] = cv2.add(img[:, :, 0], intensity)
    elif cast_type == 'red':
        img[:, :, 2] = cv2.add(img[:, :, 2], intensity)
    elif cast_type == 'green':
        img[:, :, 1] = cv2.add(img[:, :, 1], intensity)
    return img


def deep(CFA):

    #image_path = f'images/ground_truth/{name[ind]}'

    save_dir = ''
    net_name = 'model_state_dict_raw_raw.pth'
    trained_net, lossfun, optimizer= createUDnCNN()
    model_path = os.path.join(save_dir, net_name)
    trained_net.load_state_dict(torch.load(model_path))


    #rgb_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    #CFA = cv2.imread(f'images/mosaiced_noisetest/{name[ind]}')
    bayer_img = CFA_pattern(CFA,'grbg')
    


    print(bayer_img.shape)
    stride = 10
    dim = 16
    prediction = inference(bayer_img, trained_net, dim, stride)
    pred_img = np.transpose(prediction, (1, 2, 0))


    # output_path = os.path.join('outputs/output_images/phase2/demosaiced', 'deep_img.png')
    # img_normalized = cv2.normalize(pred_img, None, 0, 255, cv2.NORM_MINMAX)
    # img_uint8 = img_normalized.astype('uint8')
    # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, img_bgr)

    #rmse = calculate_rmse(rgb_img, pred_img)


    # plt.figure(figsize=(10, 15))
    # # plt.subplot(1, 3, 1)
    # # plt.imshow(rgb_img)
    # # plt.title('Ground Truth Image')

    # plt.subplot(1, 3, 2)
    # plt.imshow(bayer_img, cmap='gray')
    # plt.title('Bayer Pattern Image')

    # plt.subplot(1, 3, 3)
    # plt.imshow(pred_img)
    # plt.title('Interpolated Image')

    # plt.show()

    #print('rmse value: ', rmse)

    # img = pred_img
    # ####################### Histogram ################
    # #img_shift = np.float32(img) / 255.0 *0.3
    # img_hist = clahe_custom(img)
    
    
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # ax[0].imshow(img)
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

    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    return pred_img




def lin(CFA):
    patternB = ['Bbest-rggb','dirtyred','red_times20', 'red_V2']
    patternA = ['Abest-rggb', 'dirtygreen','green_times20', 'green_V2']
    RK = np.load(f"outputs/matrix/{patternB[0]}.npy")

    GK = np.load(f"outputs/matrix/{patternA[0]}.npy")

    print("G", GK)
    print("R", RK)
    #gbrg
    #CFA = plt.imread(f'images/mosaiced_noise/{name[ind]}')
    img = demosaic(CFA,RK,GK,norm=False, pattern='RGGB')
    return img

    # output_path = os.path.join('outputs/output_images/phase2/demosaiced/lin.png')
    # img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # img_uint8 = img_normalized.astype('uint8')
    # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, img_bgr)

    # plt.imshow(img)
    # plt.title('Demosaic Image')
    # plt.show()

    # ###################### Histogram ################
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

name = [ 'truck.png','close-up-of-tulips-blooming-in-field-royalty-free-image-1584131603.png','Colourfull_of_birds.png','20231230_150122.png','20231230_145808.png','20231230_150111.png']

CFA = plt.imread(f'images/mosaiced_noise/{name[-1]}')
CFA_DEEP = cv2.imread(f'images/mosaiced_noise/{name[-1]}')
#lin(CFA)

#deep(CFA)




############### short



#img = read_arw(f'images/sony/short/00001_00_0.ARW')

with rawpy.imread('images/sony/short/bikes.ARW') as raw:
    # Postprocess the raw data minimally
    img = raw.postprocess(no_auto_bright=True)

#img_ref = read_arw('images/sony/long/00001_00_10s.ARW')

with rawpy.imread('images/sony/long/bikes.ARW') as raw:
    # Postprocess the raw data minimally
    img_ref = raw.postprocess()

img_pred = deep(img)

# img_hist = clahe_custom(img)


fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 2 columns

ax[0].imshow(img)
ax[0].set_title('Dark')
ax[0].axis('off')  

ax[1].imshow(img_ref)
ax[1].set_title('Ground Truth')
ax[1].axis('off')

ax[2].imshow(img_pred)
ax[2].set_title('out')
ax[2].axis('off')  
plt.show()

img = img_pred
img_hist = clahe_custom(img)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax[0].imshow(img)
ax[0].set_title('Dark')
ax[0].axis('off')  

ax[1].imshow(img_hist)
ax[1].set_title('Histogram')
ax[1].axis('off')  
plt.show()

img_hist = clahe2(img)


fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

ax[0].imshow(img)
ax[0].set_title('Dark')
ax[0].axis('off')  

ax[1].imshow(img_hist)
ax[1].set_title('Histogram')
ax[1].axis('off')  
plt.show()

# img_hist = clahe_custom(img_pred)


# fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax[0].imshow(img_pred)
# ax[0].set_title('Dark')
# ax[0].axis('off')  

# ax[1].imshow(img_hist)
# ax[1].set_title('Histogram')
# ax[1].axis('off')  
# plt.show()