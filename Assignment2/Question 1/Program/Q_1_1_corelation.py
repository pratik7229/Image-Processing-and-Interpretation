import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the "Input_image" image
Input_image = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\Image.pgm').convert('L')

# Convert the PIL image to a NumPy array
Input_image_array = np.array(Input_image)

# Function to perform spatial filtering (correlation)
def spatial_filter(image, kernel):
    height, width = image.shape
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    output = np.zeros((height, width), dtype=np.uint8)

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            filtered_value = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    filtered_value += neighborhood[m, n] * kernel[m, n]
            output[i, j] = filtered_value

    return output

# Customize the filter size and values (weights)
filter_size = 7
filter_values = np.array([[1, 1, 2, 2, 2, 1, 1],
                          [1, 2, 2, 4, 2, 2, 1],
                          [2, 2, 4, 8, 4, 2, 2],
                          [2, 4, 8, 16, 8, 4, 2],
                          [2, 2, 4, 8, 4, 2, 2],
                          [1, 2, 2, 4, 2, 2, 1],
                          [1, 1, 2, 2, 2, 1, 1]])


# Normalize the filter values to ensure the sum is 1 for preserving brightness
filter_values = filter_values / np.sum(filter_values)

# Apply spatial filtering with the custom filter
filtered_Input_image = spatial_filter(Input_image_array, filter_values)

# Convert the NumPy array back to a PIL image
filtered_Input_image = Image.fromarray(filtered_Input_image)

# Display the original and filtered images
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(Input_image_array, cmap='gray')
axes[0].set_title('Original Input_image')
axes[0].axis('off')

axes[1].imshow(filtered_Input_image, cmap='gray')
axes[1].set_title('Filtered Input_image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
