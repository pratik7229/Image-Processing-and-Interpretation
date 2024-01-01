import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Load the Lenna and sf image
image_lenna = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\lenna.pgm').convert('L')  # Convert to grayscale
image_sf = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\sf.pgm').convert('L')  # Convert to grayscale
image_array_lenna = np.array(image_lenna)
image_array_sf=np.array(image_sf)

# Define the size of the Gaussian filter
filter_size = 7

# Define the Gaussian filter mask
mask = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 1, 1, 1],
])

# Get the dimensions of the image lenna and sf
height_lenna, width_lenna = image_array_lenna.shape
height_sf, width_sf = image_array_sf.shape


# Initialize an empty array for the smoothed image
smoothed_image_lenna = np.zeros((height_lenna, width_lenna), dtype=np.float32)
smoothed_image_sf = np.zeros((height_sf, width_sf), dtype=np.float32)

# Apply the Gaussian filter to each pixel in the image lenna
for i in range(height_lenna):
    for j in range(width_lenna):
        patch = image_array_lenna[max(0, i - filter_size // 2):min(height_lenna, i + filter_size // 2 + 1),
                      max(0, j - filter_size // 2):min(width_lenna, j + filter_size // 2 + 1)]
        smoothed_image_lenna[i, j] = np.sum(patch * mask[:patch.shape[0], :patch.shape[1]])

# Apply the Gaussian filter to each pixel in the image sf
for k in range(height_sf):
    for l in range(width_sf):
        patch = image_array_sf[max(0, k - filter_size // 2):min(height_sf, k + filter_size // 2 + 1),
                      max(0, l - filter_size // 2):min(width_sf, l + filter_size // 2 + 1)]
        smoothed_image_sf[k, l] = int(np.sum(patch * mask[:patch.shape[0], :patch.shape[1]]))

# Normalize the smoothed lenna image to 0-255 range
smoothed_image_lenna = ((smoothed_image_lenna - np.min(smoothed_image_lenna)) / (np.max(smoothed_image_lenna) - np.min(smoothed_image_lenna)) * 255).astype(np.uint8)

# Normalize the smoothed sf image to 0-255 range
smoothed_image_sf = ((smoothed_image_sf - np.min(smoothed_image_sf)) / (np.max(smoothed_image_sf) - np.min(smoothed_image_sf)) * 255).astype(np.uint8)

# Save the smoothed lenna image
smoothed_image_lenna = Image.fromarray(smoothed_image_lenna)
smoothed_image_lenna.save('D:\Program_jalpa\Image_processing\Asssignment_2\Images\smoothed_Gaussian_lenna.jpg')

# Save the smoothed sf image
smoothed_image_sf = Image.fromarray(smoothed_image_sf)
smoothed_image_sf.save('D:\Program_jalpa\Image_processing\Asssignment_2\Images\smoothed_Gaussian_sf.jpg')

# Create a figure with subplots for the original image, equalized image
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot the original image lenna and sf
ax1.imshow(image_array_lenna, cmap='gray')
ax1.set_title('Original lenna Image')
ax1.axis('off')

ax2.imshow(smoothed_image_lenna, cmap='gray')
ax2.set_title('Gaussian Smoothed lenna Image')
ax2.axis('off')

ax3.imshow(image_array_sf, cmap='gray')
ax3.set_title('Original sf Image')
ax3.axis('off')

ax4.imshow(smoothed_image_sf, cmap='gray')
ax4.set_title('Gaussian Smoothed sf Image')
ax4.axis('off')

plt.tight_layout()
plt.show()

