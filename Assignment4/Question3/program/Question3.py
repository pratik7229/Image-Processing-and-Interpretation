#!/usr/bin/env python
# coding: utf-8

# In[36]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(image, D0, c, gamma_L, gamma_H):
    # Convert the image to float32
    image_float = np.float32(image)

    # Apply logarithmic transformation
    log_transformed = np.log1p(image_float)

    # Perform Fourier transform
    f_transform = np.fft.fft2(log_transformed)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Create a meshgrid for u and v
    rows, cols = image.shape
    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))

    # Calculate the distance from each point to the center
    distance = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)

    # High-pass filter equation
    H_uv = (gamma_H - gamma_L) * (1 - np.exp(-c * (distance**2) / (D0**2))) + gamma_L

    # Apply the high-pass filter to the Fourier transform
    f_transform_filtered = f_transform_shifted * H_uv

    # Perform inverse Fourier transform
    inverse_transform = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)).real

    # Apply exponential to revert the log transformation
    output = np.exp(inverse_transform) - 1

    # Normalize to 8-bit for display
    output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return output_normalized

# Load the image
image_path = '/home/pratik/courses/674/program/database/pgm/girl.pgm'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Unable to load the image from {image_path}")
else:
    # Set the parameter values
    D0 = 1.8
    c = 1
    gamma_L_values = [0.0,0.25,0.75]
    gamma_H_values = [1.0,1.25,1.75]

    # Experiment with different parameter values
    plt.figure(figsize=(18, 18))
    plt.subplot(3, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.show()
    
    gamma_L = 0.5
    gamma_H = 1.5
    filtered_image = homomorphic_filter(image, D0, c, gamma_L, gamma_H)
    
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'γ_L={gamma_L:.2f}, γ_H={gamma_H:.2f}')
    plt.show()
    
    subplot_index = 1
    
    for gamma_L in gamma_L_values:
        for gamma_H in gamma_H_values:
            # Perform homomorphic filtering
            filtered_image = homomorphic_filter(image, D0, c, gamma_L, gamma_H)

            # Define the subplot layout
        
            plt.subplot(3,3, subplot_index)
            plt.imshow(filtered_image, cmap='gray')
            plt.title(f'γ_L={gamma_L:.2f}, γ_H={gamma_H:.2f}')

            subplot_index += 1

    # Display all subplots
    plt.tight_layout()
    plt.show()


# In[ ]:




