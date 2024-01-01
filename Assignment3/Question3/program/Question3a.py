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


def compute_1d_dft(signal, arg2):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    if arg2 == "F":
        e = np.exp(-2j * np.pi * k * n / N)
        return np.dot(e, signal)
    elif arg2 == "I":
        e = np.exp(2j * np.pi * k * n / N)
        return np.dot(e, signal)/N

def compute_with_zero_phase(image):
    # Compute the 1D DFT along rows using broadcasting
    dft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=image, arg2="F")

    # Compute the 1D DFT along columns using broadcasting
    dft_image = np.apply_along_axis(compute_1d_dft, axis=0, arr=dft_rows, arg2="F")

    # Calculate the magnitude manually
    magnitude = np.sqrt(dft_image.real**2 + dft_image.imag**2)

    # Create a complex array with magnitude and zero phase
    complex_array = magnitude * np.exp(0j)

    # Compute the 1D IDFT along rows using broadcasting
    idft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=complex_array, arg2="I")

    # Compute the 1D IDFT along columns using broadcasting
    result = np.apply_along_axis(compute_1d_dft, axis=0, arr=idft_rows, arg2="I")

    # Resultant image is stored in result
    result = np.sqrt(result.real**2 + result.imag**2)
    result = np.abs(result).astype(np.uint8)
    return result


def plotting1(image, result):
    # Display the original and reconstructed images
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('Reconstructed Image with Zero Phase')

    plt.show()
    


imagePath = '/home/pratik/courses/674/program/database/pgm/lenna.pgm'
image = importImage(imagePath)

result = compute_with_zero_phase(image)
plotting1(image, result)



