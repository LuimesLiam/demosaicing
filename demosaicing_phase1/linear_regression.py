from typing import Tuple
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
import math
import torch
import torch.optim as optim
import sys
import threading
import time

save_work = False
print_status = False
input_str = []
def masks_CFA_Bayer(shape: int or Tuple[int, ...], pattern: str = "RGGB") -> Tuple[np.ndarray, ...]:


    pattern = pattern.upper()
    channels = {channel: np.zeros(shape, dtype="bool") for channel in "RGB"}
    
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels.values())


def learn(input_images, kernels, ground_truths=None, gamma=0.05, name=None):
    global save_work
    global print_status
    global input_str
    for k in range(len(input_images)):
        input_height, input_width = input_images[k].shape
        kernel_height, kernel_width = kernels[k].shape

        # Calculate padding sizes
        pad_height = (kernel_height - 1) // 2
        pad_width = (kernel_width - 1) // 2

        padded_image = np.pad(input_images[k], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        input_image = torch.tensor(input_images[k], dtype=torch.float32)
        kernel = torch.tensor(kernels[k], dtype=torch.float32)

        ground_truth = torch.tensor(ground_truths[k], dtype=torch.float32)

        kernel.requires_grad = True

        output = torch.zeros_like(input_image)

        optimizer = optim.SGD([kernel], lr=gamma)


        start_index = input_height // 3
        end_index = 2 * (input_height // 3)
        start_index2 = input_width // 3
        end_index2 = 2 * (input_width // 3)
        
        for i in range(start_index, end_index):
            for j in range(start_index2, end_index2):
                region = padded_image[i:i+kernel_height, j:j+kernel_width]

                itt = 0
            
                if (save_work == True): ## to allow inputs to save or whatnot
                    print("SAVE")
                    nkernel = kernel.detach().numpy() 
                    np.save(f'outputs/matrix/dirty{name[k]}', nkernel)
                    save_work = False
                if (input_str):
                    if (input_str[0] == 'tol'):
                        tol = input_str[1]
                        print("change tol",tol)

                    if (input_str[0]=="next"):
                        i=end_index
                        j=end_index2
                        k +=1 
                        print("next", name[k])
                        
                    input_str = []


                output[i, j] = torch.sum(torch.tensor(region, dtype=torch.float32) * kernel)
                error = ground_truth[i, j] - output[i, j]
                mse = torch.mean(error ** 2)
                optimizer.zero_grad()
                mse.backward(retain_graph=True)  
                optimizer.step()

                if (print_status == True):
                    print("norm:", torch.norm(kernel.grad), j / input_width * 100, i / input_height * 100,name[k])
                    print_status = False
                    
           


        nkernel = kernel.detach().numpy()   
        print(name[k], nkernel)
        np.save(f'outputs/matrix/{name[k]}', nkernel)
    
def input_thread():
    global save_work
    global print_status
    global input_str
    while True:
        user_input = input()
        if user_input.lower() == "s":
            save_work = True
        if user_input.lower() == "p":
            print_status = True
        if "val:" in user_input: 
            ## val:count:value
            parts = user_input.split(':')
            input_str = [parts[1].strip(),float(parts[2].strip())]
        if("next" in user_input):
            input_str= [user_input, 0]
        time.sleep(0.5)
# Example usage:

print("start")
names = ['bird.png', 'truck.png','snow-dog.png']
ind = 2
CFA = plt.imread(f'images/mosaiced/{names[ind]}')
gt = plt.imread(f'images/ground_truth/{names[ind]}')

R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, "RGGB")
R = CFA * R_m
G = CFA * G_m 
B = CFA * B_m

R_gt, G_gt, B_gt = gt[:,:,0], gt[:,:,1], gt[:,:,2]


GK= np.random.rand(5,5)

RBK= np.random.rand(5,5)

thr = threading.Thread(target=input_thread, daemon=True)
thr.start()
learn([G,R,B], [GK,RBK,RBK], [G_gt,R_gt,B_gt], name=["green","red", 'blue'] )


