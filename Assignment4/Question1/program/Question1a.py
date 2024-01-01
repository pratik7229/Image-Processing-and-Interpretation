#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all the libraries here
from PIL import Image # library for importing Image
import numpy as np    # library for math operations
import matplotlib.pyplot as plt # library for displaying the image
import cv2


# In[2]:


# define function to import image

def importImage(imagePath):
    
    """ this function takes in the image path as input and returns the image in array format """
    image = Image.open(imagePath) 
    image = np.array(image) # convert opened Image to numpy array
    return image  # returns the numpy array of image


# In[10]:


def padding(image):
    M, N = image.shape
    P = next_power_of_2(2 * M)
    Q = next_power_of_2(2 * N)

    # Pad the image to the desired size
    padded_image = np.zeros((P, Q), dtype=image.dtype)
    padded_image[:M, :N] = image
    return padded_image

def centering(padded_image):
    # Multiply by (-1)^(x + y) to center the spectrum
    M, N = image.shape
    P = next_power_of_2(2 * M)
    Q = next_power_of_2(2 * N)
    x = np.arange(P)
    y = np.arange(Q)
    x, y = np.meshgrid(x, y)
    centering_multiplier = (-1) ** (x + y)
    centered_image = padded_image * centering_multiplier

    return centered_image


# In[ ]:





# In[11]:


def next_power_of_2(x):
    """
    Find the next power of 2 greater than or equal to x.
    """
    return 1 << (x - 1).bit_length()


# In[12]:


# Define a function to compute the 1D DFT
def compute_1d_dft(signal, arg2):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    if arg2 == "F":
        e = np.exp(-2j * np.pi * k * n / N)
        return np.dot(e, signal)
    elif arg2 == "I":
        e = np.exp(-2j * np.pi * k * n / N)
        return np.dot(e, signal)


# In[13]:


def computeDFT(image):
    # Compute the 1D DFT along rows using broadcasting
    dft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=image, arg2="F")
    # Compute the 1D DFT along columns using broadcasting
    dft_image = np.apply_along_axis(compute_1d_dft, axis=0, arr=dft_rows, arg2="F")
    return dft_image

def computeIDFT(G_uv):
    # Compute the 1D DFT along rows using broadcasting
    dft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=image, arg2="I")
    # Compute the 1D DFT along columns using broadcasting
    dft_image = np.apply_along_axis(compute_1d_dft, axis=0, arr=dft_rows, arg2="I")
    return dft_image


# In[21]:


def filterBandreject(P, Q):
    # Create a meshgrid for u and v
    u, v = np.meshgrid(np.fft.fftfreq(Q), np.fft.fftfreq(P))
    # Set the frequency range for the band-reject filter (you can adjust these values)
    low_freq_cutoff = 0.05
    high_freq_cutoff = 0.2
    
    # Create the band-reject filter function
    band_reject_filter_function = np.ones((P, Q), dtype=np.float64)
    band_reject_filter_function[(u**2 + v**2 >= low_freq_cutoff**2) & (u**2 + v**2 <= high_freq_cutoff**2)] = 0
    
    # Create H(u, v) with center at (P/2, Q/2)
    center_u, center_v = P // 2, Q // 2
    H_uv = np.roll(band_reject_filter_function, shift=(center_u, center_v), axis=(0, 1))
    return H_uv


# In[ ]:





# In[22]:


def complex_multiply(a, b):
    return a.real * b.real - a.imag * b.imag + 1j * (a.real * b.imag + a.imag * b.real)


# In[23]:


imagePath = '/home/pratik/courses/674/program/database/pgm/boy_noisy.pgm'
image = importImage(imagePath)
# Assuming 'image' is a grayscale image, convert it to a 2D array
if len(image.shape) == 3:
    image = np.mean(image, axis=-1)
    
# Pad the image to the desired size
paddedImage = padding(image)
centered_image = centering(paddedImage)
# f_transform = computeDFT(paddedImage)
f_transform = np.fft.fft2(centered_image)
# Set the size of the filter
P, Q = 1024, 1024

# Generate Butterworth band-reject frequency response
H_bandreject = filterBandreject(P, Q)
G_uv_shifted = complex_multiply(H_bandreject,f_transform)

plt.subplot(121), plt.imshow(paddedImage, cmap='gray')
plt.title('Padded Image'), plt.xticks([]), plt.yticks([])


gp_spatial = np.fft.ifft2(np.fft.ifftshift(G_uv_shifted)).real
gp_spatial = np.uint8(gp_spatial)


# Display the original image
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Display the filtered image
plt.subplot(122), plt.imshow(gp_spatial, cmap='gray')
plt.title('Filtered Image (g_p(x, y))'), plt.xticks([]), plt.yticks([])

M, N = image.shape
# Extract the M x N region from the top-left quadrant of gp(x, y) to undo padding
g_filtered = gp_spatial[:M, :N]

# Display the final filtered result
plt.imshow(g_filtered, cmap='gray')
plt.title('Final Filtered Result (g(x, y))'), plt.xticks([]), plt.yticks([])
plt.show()


# In[24]:


plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Noisy Image')

plt.subplot(1, 2, 2)
plt.imshow(paddedImage, cmap='gray')
plt.title('Filtered Image')

plt.show()


# In[36]:


dft_image = f_transform
magnitude = np.sqrt(dft_image.real**2 + dft_image.imag**2)
mag = np.log1p(magnitude)


plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(mag, cmap='gray')
plt.title('spectrum of orignal Image')

plt.subplot(1, 3, 2)
plt.imshow(H_bandreject , cmap='gray')
plt.title('Filter')



plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




