import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load the input image and the template image
input_image = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Image.pgm').convert('L')
template_image = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\pattern.pgm').convert('L')

# Convert images to NumPy arrays
input_array = np.array(input_image)
template_array = np.array(template_image)

# Perform template matching
template_height, template_width = template_array.shape
input_height, input_width = input_array.shape

# Initialize the result matrix to store correlation scores
result = np.zeros((input_height - template_height + 1, input_width - template_width + 1))

# Iterate through the input image and calculate correlations
for i in range(input_height - template_height + 1):
    for j in range(input_width - template_width + 1):
        patch = input_array[i:i+template_height, j:j+template_width]
        correlation = np.sum(patch * template_array)
        result[i, j] = correlation

# Find peak correlation scores
threshold = 0.9  # You can adjust this threshold as needed
peak_locations = np.where(result >= threshold * np.max(result))

# Create a copy of the input image for visualization
output_image = Image.fromarray(input_array)  # Convert to PIL image

# Draw bounding boxes around the detected patterns (at peak locations)
draw = ImageDraw.Draw(output_image)
for (row, col) in zip(*peak_locations):
    top_left = (col, row)
    bottom_right = (col + template_width - 1, row + template_height - 1)
    draw.rectangle([top_left, bottom_right], outline=255, width=1)
    
# Convert the result image to NumPy array for display
result_array = np.array(result * 255, dtype=np.uint8)

# Display the result and output images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(input_array, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(template_array, cmap='gray')
axes[1].set_title('Template Image')
axes[1].axis('off')

axes[2].imshow(output_image, cmap='gray')
axes[2].set_title('Detected Patterns')
axes[2].axis('off')

plt.tight_layout()
plt.show()
