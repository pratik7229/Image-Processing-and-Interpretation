

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


def pad_matrix(matrix):
    # Get the dimensions of the original matrix
    original_rows, original_cols = len(matrix), len(matrix[0])

    # Calculate the new dimensions after padding
    padded_rows, padded_cols = original_rows + 4, original_cols + 4  # 2 pixels on each side

    # Create a new matrix filled with zeros
    padded_matrix = [[0 for _ in range(padded_cols)] for _ in range(padded_rows)]

    # Copy the original matrix into the center of the padded matrix
    for i in range(original_rows):
        for j in range(original_cols):
            padded_matrix[i + 1][j + 1] = matrix[i][j]
    padded_matrix = np.array(padded_matrix)
    return padded_matrix



def dimensionMatching(matrix):
        # Get the dimensions of the original matrix
    shp = gaussSmoothImg.shape[0]
    diff = abs(shp - 256)
    original_rows, original_cols = len(matrix), len(matrix[0])

    # Calculate the new dimensions after padding
    padded_rows, padded_cols = original_rows + diff, original_cols + diff 

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
    template = kernel

    img_height, img_width = image.shape
    template_height, template_width = template.shape

    # Initialize a result matrix to store match scores
    result = np.zeros((img_height - template_height + 1, img_width - template_width + 1))

    for y in range(img_height - template_height + 1):
        for x in range(img_width - template_width + 1):
            # Extract a region from the image to compare with the template
            region = image[y:y + template_height, x:x + template_width]
            
            sm = np.sum(region * template)
            result[y, x] = sm

    return result



def imageOpersub(image, guassSmoothImage):
    gmask = image - guassSmoothImage
    return gmask


def imageOperAdd(image, gmask, k):
    sharpImg = image + (k*gmask)
    return sharpImg


def plotting(image):
    
    # Plot the first image on the left subplot
    plt.imshow(image, cmap='gray')


# define function to plot the image
def plottingGussian(image, guassian):
    
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(11,11))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(guassian, cmap='gray')
    axes[1].set_title('gaussian smooth Image')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



# define function to plot the image
def plottingFinalSharp(image, sharp):
    
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(11,11))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(sharp, cmap='gray')
    axes[1].set_title('sharp Image')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



def plottingGmask(image, guassian, gmask):

    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(11,11))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(guassian, cmap='gray')
    axes[1].set_title('Guassian Smooth Image')
    
    axes[2].imshow(gmask, cmap='gray')
    axes[2].set_title('Gmask')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



def plottingSharp(image, guassian, gmask,sharpImg):
    
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 4, figsize=(11,11))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(guassian, cmap='gray')
    axes[1].set_title('Guassian Smooth Image')
    
    axes[2].imshow(gmask, cmap='gray')
    axes[2].set_title('Gmask k = 4')
    
    axes[3].imshow(sharpImg, cmap='gray')
    axes[3].set_title('Sharp Image')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# parameters
sigma = 2
filter_size = 7
k = 1

# import orignal Image
imagePath = '/home/pratik/courses/674/program/database/pgm/lenna.pgm'
image= importImage(imagePath)

paddImage = pad_matrix(image)

# apply gaussian smoothning with 7x7 mask and sigma
gaussSmoothImg = operationGaussian(paddImage, filter_size, sigma)

# match the dimension using padding
dimMatchImg = dimensionMatching(gaussSmoothImg)

# subtract orignal image from smooth image and get gmask
gmask = imageOpersub(image, dimMatchImg)
gmask = np.round(gmask).astype(int)

# add orignal image with the weighted gmask and get sharp image
sharpImg = imageOperAdd(image, gmask, k)
sharpened_image = np.clip(sharpImg, 0, 255)

# Convert to uint8 type for display
sharpImg = np.uint8(sharpened_image)
# plot the orignal image 
plotting(image)

# plot the gaussian smooth image and orignal image
plottingGussian(image, gaussSmoothImg)

# plot the gmask
plottingGmask(image, gaussSmoothImg, gmask)

# plot the sharp image
plottingSharp(image, gaussSmoothImg, gmask,sharpImg)

plottingFinalSharp(image, sharpImg)


# parameters
sigma = 1
filter_size = 7
k = 2

# import orignal Image
imagePath = '/home/pratik/courses/674/program/database/pgm/lenna.pgm'
image= importImage(imagePath)

# apply gaussian smoothning with 7x7 mask and sigma 1
paddImage = pad_matrix(image)
# apply gaussian smoothning with 7x7 mask and sigma 1
gaussSmoothImg = operationGaussian(paddImage, filter_size, sigma)

dimMatchImg = dimensionMatching(gaussSmoothImg)

# subtract orignal image from smooth image and get gmask
gmask = imageOpersub(image, dimMatchImg)


# add orignal image with the weighted gmask and get sharp image
gmask = np.round(gmask).astype(int)

# add orignal image with the weighted gmask and get sharp image
sharpImg = imageOperAdd(image, gmask, k)
sharpened_image = np.clip(sharpImg, 0, 255)

# Convert to uint8 type for display
sharpImg = np.uint8(sharpened_image)

# plot the orignal image 
plotting(image)

# plot the gaussian smooth image and orignal image
plottingGussian(image, gaussSmoothImg)

# plot the gmask
plottingGmask(image, gaussSmoothImg, gmask)

# plot the sharp image
plottingSharp(image, gaussSmoothImg, gmask,sharpImg)

plottingFinalSharp(image, sharpImg)


