import numpy as np

def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter to the image manually.
    
    Parameters:
        image (numpy.ndarray): The input image as a NumPy array.
        kernel_size (int): The size of the kernel. Must be an odd number.
    
    Returns:
        numpy.ndarray: The denoised image.
    """
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

# Example usage:
if __name__ == "__main__":
    # Assuming `noised_image` is your input NumPy array
    # noised_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    # denoised_image = apply_median_filter(noised_image, kernel_size=3)
    
    # You would load your image here and convert it to a NumPy array.
    # Remember to adjust `kernel_size` based on your needs.
    pass
