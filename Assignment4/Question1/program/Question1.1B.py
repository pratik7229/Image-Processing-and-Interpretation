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


# In[3]:


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


# In[4]:


def next_power_of_2(x):
    """
    Find the next power of 2 greater than or equal to x.
    """
    return 1 << (x - 1).bit_length()


# In[5]:


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


# In[6]:


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


# In[7]:


def notch_filter(shape, center, radius):
    rows, cols = shape
    center_row, center_col = center

    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    distance = np.sqrt((u - center_col)**2 + (v - center_row)**2)

    notch_filter = np.ones((rows, cols))
    notch_filter[distance <= radius] = 0

    return notch_filter


# In[8]:


def complex_multiply(a, b):
    return a.real * b.real - a.imag * b.imag + 1j * (a.real * b.imag + a.imag * b.real)


# In[9]:


imagePath = '/home/pratik/courses/674/program/database/pgm/boy_noisy.pgm'
image = importImage(imagePath)
# Assuming 'image' is a grayscale image, convert it to a 2D array
if len(image.shape) == 3:
    image = np.mean(image, axis=-1)
    
# Pad the image to the desired size
paddedImage = padding(image)
centered_image = centering(paddedImage)


# In[10]:


plt.imshow(paddedImage,cmap='gray')
plt.title('padded Image')
plt.show()


# In[11]:


# f_transform = computeDFT(paddedImage)
f_transform = np.fft.fft2(centered_image)

dft_image = f_transform
magnitude = np.sqrt(dft_image.real**2 + dft_image.imag**2)
mag = np.log1p(magnitude)

# Set the notch filter parameters
center_frequency = (0.01, 0.5)  # Adjust the center frequency based on your requirements
radius = 1.06  # Adjust the radius based on your requirements

# Generate the notch filter
filter_shape = (1024,1024)  # Adjust the size based on your image size
notch_filter = notch_filter(filter_shape, center_frequency, radius)

plt.figure(figsize=(18, 5))
plt.imshow(mag, cmap='gray')
plt.title('spectrum of orignal Image')


plt.show()


# In[ ]:





# In[12]:


G_uv_shifted = complex_multiply(notch_filter,f_transform)

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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




