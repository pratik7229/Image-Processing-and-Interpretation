
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



import numpy as np

def salt_and_pepper_noise(image, total_noise_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Calculate the number of white (salt) and black (pepper) pixels
    num_salt = int(total_pixels * total_noise_prob / 2)
    num_pepper = int(total_pixels * total_noise_prob / 2)

    # Add salt noise (white pixels)
    salt_coordinates = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1]] = 255

    # Add pepper noise (black pixels)
    pepper_coordinates = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1]] = 0

    return noisy_image


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



def medianFiltering(image, filter_size):
    
    # Create a 7x7 median filter array
    median_filter = np.zeros((filter_size, filter_size), dtype=np.uint8)
    median_filter[filter_size // 2, filter_size // 2] = 1
    template = median_filter
    img_height, img_width = image.shape
    template_height, template_width = template.shape

    # Initialize a result matrix to store match scores
    result = np.zeros((img_height - template_height + 1, img_width - template_width + 1))
    
    # Perform template matching
    for y in range(img_height - template_height + 1):
        for x in range(img_width - template_width + 1):
            # Extract a region from the image to compare with the template
            region = image[y:y + template_height, x:x + template_width]
            # Calculate the sum of squared differences (SSD) as the match score
            median_value = np.median(region)
            result[y, x] = median_value

    return result


def averageFiltering(image, filter_size):
    
    # Create a 7x7 median filter array
    median_filter = np.zeros((filter_size, filter_size), dtype=np.uint8)
    median_filter[filter_size // 2, filter_size // 2] = 1
    template = median_filter
    img_height, img_width = image.shape
    template_height, template_width = template.shape

    # Initialize a result matrix to store match scores
    result = np.zeros((img_height - template_height + 1, img_width - template_width + 1))

    # Perform template matching
    for y in range(img_height - template_height + 1):
        for x in range(img_width - template_width + 1):
            # Extract a region from the image to compare with the template
            region = image[y:y + template_height, x:x + template_width]
            # Calculate the sum of squared differences (SSD) as the match score
            mean_value = np.mean(region)
            result[y, x] = mean_value

    return result


def plotting(image):
    
    # Plot the first image on the left subplot
    plt.imshow(image, cmap='gray')



# define function to plot the image

def plottingNoisy(orignalImage1,noisy1, noisy2):
    
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(11,11))

    axes[0].imshow(orignalImage1, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(noisy1, cmap='gray')
    axes[1].set_title('Noisy image with X=30')
    
    axes[2].imshow(noisy2, cmap='gray')
    axes[2].set_title('Noisy image with X=50')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


# define function to plot the image

def plottingAll(noisy1, filteredImg7_30, filteredImg15_30, noisy2, filteredImg7_50, filteredImg15_50):

    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 3, figsize=(11,11))

    axes[0,0].imshow(noisy1, cmap='gray')
    axes[0,0].set_title('Noisy image X=30')

    axes[0,1].imshow(filteredImg7_30, cmap='gray')
    axes[0,1].set_title('Filter 7x7')
    
    axes[0,2].imshow(filteredImg15_30, cmap='gray')
    axes[0,2].set_title('Filter 15x15')
    
    axes[1,0].imshow(noisy2, cmap='gray')
    axes[1,0].set_title('Noisy image X=50')

    axes[1,1].imshow(filteredImg7_50, cmap='gray')
    axes[1,1].set_title('Filter 7x7')
    
    axes[1,2].imshow(filteredImg15_50, cmap='gray')
    axes[1,2].set_title('Filter 15x15')

    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



imagePath = '/home/pratik/courses/674/program/database/pgm/lenna.pgm'
image= importImage(imagePath)


salt_prob = 0.30
filter_size = 7

noisyImage1 = salt_and_pepper_noise(image, salt_prob)
paddedImage = pad_matrix(noisyImage1)


filteredImage7_30 = medianFiltering(paddedImage, filter_size)
avgFilteredImage7_30 = averageFiltering(paddedImage, filter_size)

filter_size = 15
filteredImage15_30 = medianFiltering(paddedImage, filter_size)
avgFilteredImage15_30 = averageFiltering(paddedImage, filter_size)


salt_prob  = 0.50
filter_size = 7
noisyImage2 = salt_and_pepper_noise(image, salt_prob)
paddedImage = pad_matrix(noisyImage2)


filteredImage7_50 = medianFiltering(paddedImage, filter_size)
avgFilteredImage7_50 = averageFiltering(paddedImage, filter_size)


filter_size = 15
filteredImage15_50 = medianFiltering(paddedImage, filter_size)
avgFilteredImage15_50 = averageFiltering(paddedImage, filter_size)

print("orignal Image")
plotting(image)
print("Noisy Image")
plottingNoisy(image,noisyImage1,noisyImage2)
print("median filtered Image")
plottingAll(noisyImage1, filteredImage7_30, filteredImage15_30,noisyImage2, filteredImage7_50, filteredImage15_50)

print("average filtered Image")
plottingAll(noisyImage1, avgFilteredImage7_30, avgFilteredImage15_30,noisyImage2, avgFilteredImage7_50, avgFilteredImage15_50)


