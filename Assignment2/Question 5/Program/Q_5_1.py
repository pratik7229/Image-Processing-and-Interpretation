import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the Lenna and f_16 image
image_lenna = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\lenna.pgm').convert('L')  # Convert to grayscale
image_sf = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\sf.pgm').convert('L')  # Convert to grayscale
image_array_lenna = np.array(image_lenna)
image_array_sf=np.array(image_sf)

# Define Prewitt masks
prewitt_x = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
prewitt_y = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

# Apply Prewitt masks for sharpening on lenna and sf images
lenna_prewitt_x = np.zeros_like(image_array_lenna, dtype=float)
lenna_prewitt_y = np.zeros_like(image_array_lenna, dtype=float)

sf_prewitt_x = np.zeros_like(image_array_sf, dtype=float)
sf_prewitt_y = np.zeros_like(image_array_sf, dtype=float)

# Perform manual convolution on lenna and sf images
for i in range(1, image_array_lenna.shape[0] - 1):
    for j in range(1, image_array_lenna.shape[1] - 1):
        lenna_prewitt_x[i, j] = np.sum(prewitt_x * image_array_lenna[i-1:i+2, j-1:j+2])
        lenna_prewitt_y[i, j] = np.sum(prewitt_y * image_array_lenna[i-1:i+2, j-1:j+2])

        sf_prewitt_x[i, j] = np.sum(prewitt_x * image_array_sf[i-1:i+2, j-1:j+2])
        sf_prewitt_y[i, j] = np.sum(prewitt_y * image_array_sf[i-1:i+2, j-1:j+2])

sharpened_lenna_prewitt = image_array_lenna + lenna_prewitt_x + lenna_prewitt_y
sharpened_sf_prewitt = image_array_sf + sf_prewitt_x + sf_prewitt_y

# Define Sobel masks
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Apply Sobel masks for sharpening lenna and sf images
lenna_sobel_x = np.zeros_like(image_array_lenna, dtype=float)
lenna_sobel_y = np.zeros_like(image_array_lenna, dtype=float)

sf_sobel_x = np.zeros_like(image_array_sf, dtype=float)
sf_sobel_y = np.zeros_like(image_array_sf, dtype=float)

# Perform manual convolution on lenna and sf images
for i in range(1, image_array_lenna.shape[0] - 1):
    for j in range(1, image_array_lenna.shape[1] - 1):
        lenna_sobel_x[i, j] = np.sum(sobel_x * image_array_lenna[i-1:i+2, j-1:j+2])
        lenna_sobel_y[i, j] = np.sum(sobel_y * image_array_lenna[i-1:i+2, j-1:j+2])

        sf_sobel_x[i, j] = np.sum(sobel_x * image_array_sf[i-1:i+2, j-1:j+2])
        sf_sobel_y[i, j] = np.sum(sobel_y * image_array_sf[i-1:i+2, j-1:j+2])

sharpened_lenna_sobel = image_array_lenna + lenna_sobel_x + lenna_sobel_y
sharpened_sf_sobel = image_array_sf + sf_sobel_x + sf_sobel_y

# Define Laplacian mask
laplacian_mask = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])

# Apply Laplacian mask for sharpening lenna and sf images
sharpened_lenna_laplacian = np.zeros_like(image_array_lenna, dtype=float)
sharpened_sf_laplacian = np.zeros_like(image_array_sf, dtype=float)

# Perform manual convolution on lenna and sf images
for i in range(1, image_array_lenna.shape[0] - 1):
    for j in range(1, image_array_lenna.shape[1] - 1):
        sharpened_lenna_laplacian[i, j] = np.sum(laplacian_mask * image_array_lenna[i-1:i+2, j-1:j+2])
        sharpened_sf_laplacian[i, j] = np.sum(laplacian_mask * image_array_sf[i-1:i+2, j-1:j+2])

#add sharpen image by laplacian to original lenna and sf image for result
# sharpened_lenna_laplacian = image_array_lenna + sharpened_lenna_laplacian
# sharpened_sf_laplacian = image_array_sf + sharpened_sf_laplacian

# Create a figure with subplots for the original image, equalized image
fig, axes = plt.subplots(2, 4, figsize=(12, 10))

# Plot the original image lenna and sf
axes[0, 0].imshow(image_array_lenna, cmap='gray')
axes[0, 0].set_title('Original lenna Image')
axes[0, 0].axis('off')

# Display or save the sharpened images by prewitt, sobel, laplacian
axes[0, 1].imshow(sharpened_lenna_prewitt, cmap='gray')
axes[0, 1].set_title('Sharpened Lenna (Prewitt)')
axes[0, 1].axis('off')

axes[0, 2].imshow(sharpened_lenna_sobel, cmap='gray')
axes[0, 2].set_title('Sharpened Lenna (Sobel)')
axes[0, 2].axis('off')

axes[0, 3].imshow(sharpened_lenna_laplacian, cmap='gray')
axes[0, 3].set_title('Sharpened Lenna (Laplacian)')
axes[0, 3].axis('off')

# Plot the original image lenna and sf
axes[1, 0].imshow(image_array_sf, cmap='gray')
axes[1, 0].set_title('Original sf Image')
axes[1, 0].axis('off')

# Display or save the sharpened images by prewitt, sobel, laplacian
axes[1, 1].imshow(sharpened_sf_prewitt, cmap='gray')
axes[1, 1].set_title('Sharpened SF (Prewitt)')
axes[1, 1].axis('off')

axes[1, 2].imshow(sharpened_sf_sobel, cmap='gray')
axes[1, 2].set_title('Sharpened SF (Sobel)')
axes[1, 2].axis('off')

axes[1, 3].imshow(sharpened_sf_laplacian, cmap='gray')
axes[1, 3].set_title('Sharpened SF (Laplacian)')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()