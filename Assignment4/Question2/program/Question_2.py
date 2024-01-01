import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_sobel_kernel(size):
    sobel_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    # Pad the kernel with zeros
    padded_kernel = np.pad(sobel_kernel, ((1, 1), (1, 1)), mode='constant')

    # Add a leading row and column of 0's
    padded_kernel = np.pad(padded_kernel, ((1, 0), (1, 0)), mode='constant')

    return padded_kernel

def generate_filter_transfer_function(image, kernel):
    # Create padded image and kernel
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    padded_kernel = generate_sobel_kernel(padded_image.shape)  # Ensure the kernel size matches the padded image size

    # Compute the forward DFT of the padded image and kernel
    image_fft = np.fft.fft2(padded_image)
    kernel_fft = np.fft.fft2(padded_kernel, s=padded_image.shape)  # Use 's' parameter to specify the size

    # Multiply the spectra taking care of the complex conjugate
    H_uv = image_fft * np.conj(kernel_fft)

    return H_uv

def apply_filter_frequency_domain(image, H_uv):
    # Compute the inverse DFT of the product to obtain the filtered image
    filtered_image = np.real(np.fft.ifft2(H_uv))

    return filtered_image

def apply_filter_spatial_domain(image, sobel_x):
    # Generate the padded kernel
    padded_kernel = generate_sobel_kernel(image.shape)

    # Apply Sobel masks for sharpening
    lenna_sobel_x = np.zeros_like(image, dtype=float)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            lenna_sobel_x[i, j] = np.sum(sobel_x * image[i-1:i+2, j-1:j+2])

    # Normalize the spatial domain result
    lenna_sobel_x = lenna_sobel_x / np.sum(np.abs(padded_kernel))

    return lenna_sobel_x

# Load the Lenna image
lena_path = "D:\Program_jalpa\Image_processing\Assignment_4\Images\lenna.pgm"  # Replace with the actual path to the Lena image
lena = Image.open(lena_path).convert("L")  # Convert to grayscale if needed
image = np.array(lena)

# Define Sobel masks
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Step 1: Generate Sobel kernel and filter transfer function in frequency domain
sobel_kernel = generate_sobel_kernel(image.shape)
H_uv = generate_filter_transfer_function(image, sobel_kernel)

# Step 2: Apply the filter in the frequency domain
filtered_image_freq_domain = apply_filter_frequency_domain(image, H_uv)

# Step 3: Apply the filter in the spatial domain
lenna_sobel_x = apply_filter_spatial_domain(image, sobel_x)

# Display the original image, filtered image in frequency domain, and filtered image in spatial domain
plt.figure(figsize=(12, 8))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Lenna Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(filtered_image_freq_domain, cmap='gray')
plt.title('Filtered Image (Frequency Domain)')
plt.axis('off')

plt.subplot(133)
plt.imshow(lenna_sobel_x, cmap='gray')
plt.title('Filtered Image (Spatial Domain)')
plt.axis('off')

plt.tight_layout()
plt.show()
