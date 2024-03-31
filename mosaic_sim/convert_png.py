from PIL import Image
import os

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

# Example usage:
input_directory = 'images/jpg'
output_directory = 'images/ground_truth'

convert_to_png(input_directory, output_directory)