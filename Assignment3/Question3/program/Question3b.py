
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



def compute_with_magnitude_one(image):
        # Compute the 1D DFT along rows using broadcasting
    dft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=image, arg2="F")

    # Compute the 1D DFT along columns using broadcasting
    dft_image = np.apply_along_axis(compute_1d_dft, axis=0, arr=dft_rows, arg2="F")
    rounded_array = np.round(dft_image, decimals=4)

    # Remove the negative sign for zero values
    for i in range(rounded_array.shape[0]):
        for j in range(rounded_array.shape[1]):
            if rounded_array[i, j] == -0j:
                rounded_array[i, j] = 0j
     
    magnitude = np.sqrt(rounded_array.real**2 + rounded_array.imag**2)
    
    real_part = rounded_array.real
    imaginary_part = rounded_array.imag

    phase = np.arctan2(imaginary_part, real_part)
    real_part = np.cos(phase)
    imaginary_part = np.sin(phase)
    complex_array = real_part + 1j * imaginary_part


    idft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=complex_array, arg2="I")

    # Compute the 1D IDFT along columns using broadcasting
    result = np.apply_along_axis(compute_1d_dft, axis=0, arr=idft_rows, arg2="I")



    restored_image = cv2.normalize(result.real, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return restored_image



def plotting2(image, result):
    # Display the original and reconstructed images
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('Reconstructed Image with magnitude one')

    plt.show()


imagePath = '/home/pratik/courses/674/program/database/pgm/lenna.pgm'
image = importImage(imagePath)

result = compute_with_magnitude_one(image)
plotting2(image, result)



