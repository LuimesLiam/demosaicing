from PIL import Image
import os
import shutil


def convert_to_png(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    files = os.listdir(input_dir)

    # Loop through each file
    for file in files:
        # Check if file is an image
        if file.endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            try:
                # Open image file
                image_path = os.path.join(input_dir, file)
                img = Image.open(image_path)
                
                # Convert to PNG format
                png_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.png')
                img.save(png_path, 'PNG')
                
                print(f"Converted {file} to PNG format.")
            except Exception as e:
                print(f"Error converting {file}: {e}")

# # Example usage:
# input_directory = 'images/jpg'
# output_directory = 'images/ground_truth'

# convert_to_png(input_directory, output_directory)


def copy_images(source_dir, dest_dir, num_images):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get list of image files in the source directory
    image_files = os.listdir(source_dir)

    # Take only the first 'num_images' images
    image_files = image_files[:num_images]

    # Copy corresponding images from source to destination directory
    for image_file in image_files:
        source_path = os.path.join(source_dir, image_file)
        dest_path = os.path.join(dest_dir, image_file)
        shutil.copyfile(source_path, dest_path)

# Paths to the source directories
short_dir = "C:/Users/liaml/Desktop/imageproc/Sony/short"
long_dir = "C:/Users/liaml/Desktop/imageproc/Sony/long"

# Paths to the destination directories
short_dest_dir = 'images/sony/short'
long_dest_dir = 'images/sony/long'

# Number of images to copy
num_images = 100

# Copy images from short directory
copy_images(short_dir, short_dest_dir, num_images)

# Copy images from long directory
copy_images(long_dir, long_dest_dir, num_images)

print("Images copied successfully!")