#import all the libraries here
from PIL import Image # library for importing Image
import numpy as np    # library for math operations
import matplotlib.pyplot as plt # library for displaying the image

# define function to import image

def importImage(imagePath):
    
    """ this function takes in the image path as input and returns the image in array format """
    image = Image.open(imagePath) 
    image = np.array(image) # convert opened Image to numpy array
    return image  # returns the numpy array of image


# defination of quantization logic
def quantization(image, quantizationFactor):
    """ Main Logic"""
    
    # make array of zeros with same shape as orignal Image
    qunatizedImageArray = np.zeros((image.shape[0],image.shape[1]),dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            qunatizedImageArray[i][j] = (image[i][j] / 256) * quantizationFactor
    
    # return Quantized 
    return qunatizedImageArray


# define function to plot the image

def plotting(orignalImage,quantizedImage1 ,quantizedImage2,quantizedImage3,quantizedImage4):
    """ Logic For Plotting the orignal Image and Qunatized image"""
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 5, figsize=(11,11))

    # Plot the first image on the left subplot
    axes[0].imshow(orignalImage, cmap='gray')
    axes[0].set_title('Orignal Image')

    # Plot the second image on the right subplot
    axes[1].imshow(quantizedImage1, cmap='gray')
    axes[1].set_title('Quantized Image L=128')
    
    axes[2].imshow(quantizedImage2, cmap='gray')
    axes[2].set_title('Quantized Image L=32')
    
    axes[3].imshow(quantizedImage3, cmap='gray')
    axes[3].set_title('Quantized Image L=8')
    
    axes[4].imshow(quantizedImage4, cmap='gray')
    axes[4].set_title('Quantized Image L=2')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# code Starts Here

#defination of parameters
qunatizationFactor = 4
imagePath = '/home/pratik/courses/674/program/database/pgm/peppers.pgm' # Image Path

# calling the import Image Function
image = importImage(imagePath)

# calling the Qunatization Function
quantizedImage1 = quantization(image, 128)

quantizedImage2 = quantization(image, 32)

quantizedImage3 = quantization(image, 8)

quantizedImage4 = quantization(image, 2)


plotting(image,quantizedImage1 ,quantizedImage2,quantizedImage3,quantizedImage4)
    

