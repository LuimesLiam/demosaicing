import numpy as np

def apply_median_filter(image, kernel_size=3):
    # Ensure kernel_size is odd to have a center pixel
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    # Pad the image to handle the edges
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')
    
    # Prepare the output image
    denoised_image = np.zeros_like(image)
    
    # Process each channel separately (assuming a color image)
    for channel in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extract the current kernel
                kernel = padded_image[i:i+kernel_size, j:j+kernel_size, channel]
                
                # Apply the median filter
                denoised_image[i, j, channel] = np.median(kernel)
    
    return denoised_image
