import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_black_image():
    # Create a black image
    image = np.zeros((512, 512), dtype=np.uint8)

    # Draw a white square in the center
    square_size = 32
    center_x, center_y = 256, 256
    image[center_x - square_size // 2: center_x + square_size // 2, center_y - square_size // 2: center_y + square_size // 2] = 255
    return image


# Define a function to compute the 1D DFT
def compute_1d_dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)


def computeDFT(image):
    # Compute the 1D DFT along rows using broadcasting
    dft_rows = np.apply_along_axis(compute_1d_dft, axis=1, arr=image)

    # Compute the 1D DFT along columns using broadcasting
    dft_image = np.apply_along_axis(compute_1d_dft, axis=0, arr=dft_rows)
    magnitude = np.sqrt(dft_image.real**2 + dft_image.imag**2)
    return magnitude


def compute_center(image, magnitude):
    M, N = image.shape
    center_x = N // 2
    center_y = M // 2

    # Manually shift the magnitude to the center
    magnitude_shifted = np.zeros_like(magnitude)
    magnitude_shifted[center_y:, center_x:] = magnitude[:center_y, :center_x]
    magnitude_shifted[:center_y, :center_x] = magnitude[center_y:, center_x:]
    magnitude_shifted[center_y:, :center_x] = magnitude[:center_y, center_x:]
    magnitude_shifted[:center_y, center_x:] = magnitude[center_y:, :center_x]

    # Apply the log transformation
    
    return magnitude_shifted



def applyTransformation(magnitude_shifted):
    magnitude_log = np.log(1 + magnitude_shifted)
    return magnitude_log


# define function to plot the image

def plotting2(orignalImage, magnitude_log, magnitude_shifted):
    
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(11,11))

    axes[0].imshow(orignalImage, cmap='gray')
    axes[0].set_title('Orignal Image')

    axes[1].imshow(magnitude_shifted, cmap='gray')
    axes[1].set_title('magnitude without shifting')
    
    axes[2].imshow(magnitude_log, cmap='gray')
    axes[2].set_title('centered magnitude')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


image = create_black_image()
magnitude = computeDFT(image)
magnitude_shifted = compute_center(image, magnitude)
magnitude_log = applyTransformation(magnitude_shifted)
plotting2(image, magnitude_log, magnitude_shifted)



