import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(imageA, imageB):
    # Ensure the images are in floating point in case they are in uint8
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)

    # Compute the mean squared error between the two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

CFA = plt.imread(f'images/results/demosaiced_truck-matlab.png')
gt = plt.imread(f'images/ground_truth/truck.png')

mse = mean_squared_error(gt, CFA)

print(mse)
