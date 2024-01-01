#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import all the libraries here
from PIL import Image # library for importing Image
import numpy as np    # library for math operations
import matplotlib.pyplot as plt # library for displaying the image
import cv2


# define function to import image

def importImage(imagePath):
    
    """ this function takes in the image path as input and returns the image in array format """
    image = Image.open(imagePath) 
    image = np.array(image) # convert opened Image to numpy array
    return image  # returns the numpy array of image



def pad_matrix(matrix):
    # Get the dimensions of the original matrix
    original_rows, original_cols = len(matrix), len(matrix[0])

    # Calculate the new dimensions after padding
    padded_rows, padded_cols = original_rows + 2, original_cols + 2  # 2 pixels on each side

    # Create a new matrix filled with zeros
    padded_matrix = [[0 for _ in range(padded_cols)] for _ in range(padded_rows)]

    # Copy the original matrix into the center of the padded matrix
    for i in range(original_rows):
        for j in range(original_cols):
            padded_matrix[i + 1][j + 1] = matrix[i][j]
    padded_matrix = np.array(padded_matrix)
    return padded_matrix



def operationGaussian(image, kernel_size, sigma):
    # Calculate the kernel size (should be odd)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")
    
    # Create an empty kernel
    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate the center position of the kernel
    center = kernel_size // 2

    # Calculate the sum for normalization
    total = 0

    # Fill the kernel with Gaussian values
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
            total += kernel[x, y]

    # Normalize the kernel
    kernel /= total

    # Get the image dimensions
    height, width = image.shape

    # Create an output image
    output_image = np.zeros((height, width), dtype=np.uint8)

    # Convolve the image with the kernel
    for i in range(center, height - center):
        for j in range(center, width - center):
            region = image[i - center:i + center + 1, j - center:j + center + 1]
            output_image[i, j] = int(np.sum(region * kernel))

    return output_image


# Function for operation on image
def operationAveraging(image,filter_size):
    
    
    # Create the 15x15 average filter as a NumPy array
    template = np.ones((filter_size, filter_size)) / (filter_size * filter_size)

    image_height, image_width = image.shape
    template_height, template_width = template.shape

    smoothed_image = np.zeros_like(image, dtype=np.uint8)

    # Calculate border to avoid boundary issues
    border = filter_size // 2

    for i in range(border, image_height - border):
        for j in range(border, image_width - border):
            region = image[i-border:i+border+1, j-border:j+border+1]
            smoothed_image[i, j] = int(np.sum(region * template))
    smoothed_image = smoothed_image[border:-border, border:-border]
    
    return smoothed_image



# define function to plot the image

def plottingAll(orignalImage1, smoothImg7x7, smoothImg15x15):
    """ Logic For Plotting the orignal Image and Qunatized image"""
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 3, figsize=(11,11))

    axes[0,0].imshow(orignalImage1, cmap='gray')
    axes[0,0].set_title('Orignal Image')

    axes[0,1].imshow(smoothImg7x7, cmap='gray')
    axes[0,1].set_title('Smooth Image by 7x7')
    
    axes[0,2].imshow(smoothImg15x15, cmap='gray')
    axes[0,2].set_title('Smooth Image by 15x15')

    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()




def plotting(image):
    """ Logic For Plotting the orignal Image and resized sub sampled image"""
    # Plot the first image on the left subplot
    plt.imshow(image, cmap='gray')


# this is for gaussian Filter

filter_size = 7
sigma = 2

imagePath = '/home/pratik/courses/674/program/database/pgm/boy_noisy.pgm'
image1 = importImage(imagePath)

paddedImage = pad_matrix(image1)

smoothImage7x7 = operationGaussian(paddedImage, filter_size, sigma)

filter_size = 15
smoothImage15x15 = operationGaussian(paddedImage, filter_size, sigma)




plottingAll(image1, smoothImage7x7, smoothImage15x15)


# In[ ]:




