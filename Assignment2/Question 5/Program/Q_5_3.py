import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the sf image
image_sf = Image.open('D:\Program_jalpa\Image_processing\Asssignment_2\Images\sf.pgm').convert('L')  # Convert to grayscale
image_array_sf = np.array(image_sf)

# Define Prewitt masks
prewitt_x = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]])
prewitt_y = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

# Calculate the partial derivatives in the x and y directions
sf_prewitt_x = np.zeros_like(image_array_sf, dtype=float)
sf_prewitt_y = np.zeros_like(image_array_sf, dtype=float)

for i in range(1, image_array_sf.shape[0] - 1):
    for j in range(1, image_array_sf.shape[1] - 1):
        sf_prewitt_x[i, j] = np.sum(prewitt_x * image_array_sf[i-1:i+2, j-1:j+2])
        sf_prewitt_y[i, j] = np.sum(prewitt_y * image_array_sf[i-1:i+2, j-1:j+2])

# Calculate the gradient magnitude using prewitt
sf_gradient_magnitude_prewitt = np.sqrt(sf_prewitt_x**2 + sf_prewitt_y**2)

# Define Sobel masks
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Apply Sobel masks for sharpening sf
sf_sobel_x = np.zeros_like(image_array_sf, dtype=float)
sf_sobel_y = np.zeros_like(image_array_sf, dtype=float)

# Perform manual convolution on sf
for i in range(1, image_array_sf.shape[0] - 1):
    for j in range(1, image_array_sf.shape[1] - 1):
        sf_sobel_x[i, j] = np.sum(sobel_x * image_array_sf[i-1:i+2, j-1:j+2])
        sf_sobel_y[i, j] = np.sum(sobel_y * image_array_sf[i-1:i+2, j-1:j+2])

# Calculate the gradient magnitude using sobel
sf_gradient_magnitude_sobel = np.sqrt(sf_sobel_x**2 + sf_sobel_y**2)

# Create subplots to display the results
fig, axes = plt.subplots(2, 4, figsize=(12, 8))

# Original sf Image
axes[0, 0].imshow(image_array_sf, cmap='gray')
axes[0, 0].set_title('Original Image sf')
axes[0, 0].axis('off')

# Partial Derivative in X-direction (Prewitt)
axes[0, 1].imshow(sf_prewitt_x, cmap='gray')
axes[0, 1].set_title('Partial Derivative X (Prewitt)')
axes[0, 1].axis('off')

# Partial Derivative in Y-direction (Prewitt)
axes[0, 2].imshow(sf_prewitt_y, cmap='gray')
axes[0, 2].set_title('Partial Derivative Y (Prewitt)')
axes[0, 2].axis('off')

# Gradient Magnitude (Prewitt)
axes[0, 3].imshow(sf_gradient_magnitude_prewitt, cmap='gray')
axes[0, 3].set_title('Gradient Magnitude (Prewitt)')
axes[0, 3].axis('off')

# Original sf Image
axes[1, 0].imshow(image_array_sf, cmap='gray')
axes[1, 0].set_title('Original Image sf')
axes[1, 0].axis('off')

# Partial Derivative in X-direction (Sobel)
axes[1, 1].imshow(sf_prewitt_x, cmap='gray')
axes[1, 1].set_title('Partial Derivative X (Sobel)')
axes[1, 1].axis('off')

# Partial Derivative in Y-direction (Sobel)
axes[1, 2].imshow(sf_prewitt_y, cmap='gray')
axes[1, 2].set_title('Partial Derivative Y (Sobel)')
axes[1, 2].axis('off')

# Gradient Magnitude (Sobel)
axes[1, 3].imshow(sf_gradient_magnitude_sobel, cmap='gray')
axes[1, 3].set_title('Gradient Magnitude (Sobel)')
axes[1, 3].axis('off')

# Show the plots
plt.tight_layout()
plt.show()
