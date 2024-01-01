import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the lenna image
image_lenna = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\lenna.pgm').convert('L')  # Convert to grayscale
image_array_lenna = np.array(image_lenna)

# Define Prewitt masks
prewitt_x = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
prewitt_y = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

# Calculate the partial derivatives in the x and y directions
lenna_prewitt_x = np.zeros_like(image_array_lenna, dtype=float)
lenna_prewitt_y = np.zeros_like(image_array_lenna, dtype=float)

for i in range(1, image_array_lenna.shape[0] - 1):
    for j in range(1, image_array_lenna.shape[1] - 1):
        lenna_prewitt_x[i, j] = np.sum(prewitt_x * image_array_lenna[i-1:i+2, j-1:j+2])
        lenna_prewitt_y[i, j] = np.sum(prewitt_y * image_array_lenna[i-1:i+2, j-1:j+2])

# Calculate the gradient magnitude using prewitt
lenna_gradient_magnitude_prewitt = np.sqrt(lenna_prewitt_x**2 + lenna_prewitt_y**2)

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

# Perform manual convolution on lenna and sf images
for i in range(1, image_array_lenna.shape[0] - 1):
    for j in range(1, image_array_lenna.shape[1] - 1):
        lenna_sobel_x[i, j] = np.sum(sobel_x * image_array_lenna[i-1:i+2, j-1:j+2])
        lenna_sobel_y[i, j] = np.sum(sobel_y * image_array_lenna[i-1:i+2, j-1:j+2])

# Calculate the gradient magnitude using sobel
lenna_gradient_magnitude_sobel = np.sqrt(lenna_sobel_x**2 + lenna_sobel_y**2)

# Create subplots to display the results
fig, axes = plt.subplots(2, 4, figsize=(12, 8))

# Original Lenna Image
axes[0, 0].imshow(image_array_lenna, cmap='gray')
axes[0, 0].set_title('Original Image Lenna')
axes[0, 0].axis('off')

# Partial Derivative in X-direction (Prewitt)
axes[0, 1].imshow(lenna_prewitt_x, cmap='gray')
axes[0, 1].set_title('Partial Derivative X (Prewitt)')
axes[0, 1].axis('off')

# Partial Derivative in Y-direction (Prewitt)
axes[0, 2].imshow(lenna_prewitt_y, cmap='gray')
axes[0, 2].set_title('Partial Derivative Y (Prewitt)')
axes[0, 2].axis('off')

# Gradient Magnitude (Prewitt)
axes[0, 3].imshow(lenna_gradient_magnitude_prewitt, cmap='gray')
axes[0, 3].set_title('Gradient Magnitude (Prewitt)')
axes[0, 3].axis('off')

# Original Lenna Image
axes[1, 0].imshow(image_array_lenna, cmap='gray')
axes[1, 0].set_title('Original Image Lenna')
axes[1, 0].axis('off')

# Partial Derivative in X-direction (Sobel)
axes[1, 1].imshow(lenna_prewitt_x, cmap='gray')
axes[1, 1].set_title('Partial Derivative X (Sobel)')
axes[1, 1].axis('off')

# Partial Derivative in Y-direction (Sobel)
axes[1, 2].imshow(lenna_prewitt_y, cmap='gray')
axes[1, 2].set_title('Partial Derivative Y (Sobel)')
axes[1, 2].axis('off')

# Gradient Magnitude (Sobel)
axes[1, 3].imshow(lenna_gradient_magnitude_sobel, cmap='gray')
axes[1, 3].set_title('Gradient Magnitude (Sobel)')
axes[1, 3].axis('off')

# Show the plots
plt.tight_layout()
plt.show()
