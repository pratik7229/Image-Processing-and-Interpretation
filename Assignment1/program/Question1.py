
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



# define function to sub sample the Image

def subSampling(image,subSamplingFactor):
    
    """main logic for sub Sampling"""
    
    # this is for calculating the shape of new Image
    subRows = image.shape[0] // subSamplingFactor
    subCols = image.shape[1] // subSamplingFactor
    
    # create an array of zeros 
    subSampled_Image = np.zeros((subRows, subCols), dtype=int)
    
    # sub Sample the image
    for i in range(subRows):
        for j in range(subCols):
            subSampled_Image[i, j] = image[i * subSamplingFactor, j * subSamplingFactor]
    
    # returns the sub sampled image array
    return subSampled_Image



#define function to resize the Image

def resizeImage(image,subSamplingFactor):
    
    """ Logic for Resizing the Image"""
    
    # get the height and width of the image
    height, width = image.shape
    new_height = height * subSamplingFactor
    new_width = width * subSamplingFactor
    
    # create array of zeros 
    resized_image = np.zeros((new_height, new_width), dtype=int)
    
    # fill that array of zeros with the Image pixel values
    for i in range(new_height):
        for j in range(new_width):
            source_i = i // subSamplingFactor
            source_j = j // subSamplingFactor
            resized_image[i][j] = image[source_i][source_j]
    
    # return the resized image
    return resized_image 


# define function to plot the image

def plotting(image,subSampledImage1,subSampledImage2,subSampledImage3):
    """ Logic For Plotting the orignal Image and resized sub sampled image"""
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 4, figsize=(10,10))

    # Plot the first image on the left subplot
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orignal Image')

    # Plot the second image on the right subplot
    axes[1].imshow(subSampledImage1, cmap='gray')
    axes[1].set_title('Sub Sampled by 2 Image')
    
    axes[2].imshow(subSampledImage2, cmap='gray')
    axes[2].set_title('Sub Sampled by 4 Image')
    
    axes[3].imshow(subSampledImage3, cmap='gray')
    axes[3].set_title('Sub Sampled by 8 Image')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    

# code Starts Here

# image = np.array([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9,19,11,12],
#     [13,14,15,16]
# ], dtype=np.uint8)

# defination of all the needed Parameter
subSamplingFactor = 2 # sub sampling factor variable
imagePath = '/home/pratik/courses/674/program/database/pgm/peppers.pgm' # Image Path

# returns Image in Array Format
image = importImage(imagePath)

# retruns sub Sampled Image
subSampledImage = subSampling(image, subSamplingFactor)

# retruns resized sub sampled image
subSampledImage1 = resizeImage(subSampledImage,subSamplingFactor) 

subSamplingFactor = 4 # sub sampling factor variable
imagePath = '/home/pratik/courses/674/program/database/pgm/peppers.pgm' # Image Path

# returns Image in Array Format
image = importImage(imagePath)

# retruns sub Sampled Image
subSampledImage = subSampling(image, subSamplingFactor)

# retruns resized sub sampled image
subSampledImage2 = resizeImage(subSampledImage,subSamplingFactor) 

subSamplingFactor = 8 # sub sampling factor variable
imagePath = '/home/pratik/courses/674/program/database/pgm/peppers.pgm' # Image Path

# returns Image in Array Format
image = importImage(imagePath)

# retruns sub Sampled Image
subSampledImage = subSampling(image, subSamplingFactor)

# retruns resized sub sampled image
subSampledImage3 = resizeImage(subSampledImage,subSamplingFactor) 

# Plotting the image
plotting(image,subSampledImage1,subSampledImage2,subSampledImage3)


