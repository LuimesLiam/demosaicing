import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2
from dataset_prep import Create_Dataset, CFA_pattern
import matplotlib.pyplot as plt



def learn(input_images, kernels, ground_truths=None, gamma=0.000003, name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_tensors = [torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) for img in input_images]

    ground_truth_tensors = [torch.tensor(gt, dtype=torch.float32).to(device) for gt in ground_truths]
    
    kernels_tensors = [torch.tensor(k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) for k in kernels]
    for j in range(2):
        for i, (input_tensor, kernel_tensor, ground_truth_tensor) in enumerate(zip(input_tensors, kernels_tensors, ground_truth_tensors)):
            kernel_tensor.requires_grad = True
            
            optimizer = optim.SGD([kernel_tensor], lr=gamma)
            
            for epoch in range(20000):
                optimizer.zero_grad()
                
                output = F.conv2d(input_tensor, kernel_tensor, padding='same')
                output = output.squeeze(0).squeeze(0)
                
 
                gt_for_loss = ground_truth_tensor[:, :,j ]  
                
                loss = F.mse_loss(output, gt_for_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}, {j}")
 
                if (loss.item() < 2):
                    break
            # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            # ax[0].imshow(output.detach().cpu().numpy())
            # ax[0].set_title("Output Image")
            # ax[1].imshow(gt_for_loss.cpu().numpy())
            # ax[1].set_title("Ground Truth Image")
            # plt.show()
        optimized_kernel = kernel_tensor.detach().cpu().numpy().squeeze()
        np.save(f'outputs/matrix/{name[j]}', optimized_kernel)
        print("DONE")


names = ['truck.png','Colourfull_of_birds.png','20231230_150111.png','Birthday-Flowers-Colors.jpg.png','istockphoto-696478774-612x612.png']

ground_truths = []
input_images = []

for i in names:

    image_path = f'images/ground_truth/{i}'

    rgb_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = np.array(rgb_img)

    CFA = cv2.imread(f'images/mosaiced_noise/{i}')#, cv2.IMREAD_GRAYSCALE)
    CFA =  CFA_pattern(CFA).astype(np.float32)

    ground_truths.append(rgb_img)
    input_images.append(CFA)


RK = np.load(f"outputs/matrix/Bbest-rggb.npy")

GK = np.load(f"outputs/matrix/Abest-rggb.npy")

kernels = [RK,GK,RK]#[np.random.rand(5,5), np.random.rand(5,5), np.random.rand(5,5)] 
print(kernels)

learn(input_images, kernels, ground_truths, name=["red", "green", "blue"])
