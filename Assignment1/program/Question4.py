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



def calculate_histogram(image):
    #takes in image array and returns the histogram 
    histogram = {}
    for row in image:
        for pixel in row:
            if pixel in histogram:
                histogram[pixel] += 1
            else:
                histogram[pixel] = 1
    return histogram



def calculate(histogram):
    
    pdf = {}
    pdf_value = 0
    cdf = {}
    cdf_value = 0
    total_pixels = 0
    hist_equalized = {}
    
    # calculate n value n = total number of pixels
    for value in histogram.values():
        total_pixels += value
        
    # calculate nk/n or pdf for each pixel frequency
    for intensity, frequency in sorted(histogram.items()):
        pdf[intensity] = frequency/ total_pixels
        pdf_value += frequency    
    
    # calculate sk or cdf for each pdf
    for key, value in pdf.items():
        cdf_value += value  # Add the current value to the cumulative sum
        cdf[key] = cdf_value  # Store the cumulative sum in the new dictionary
    
    # calculate sk or cdf * total number of pixels
    for key,value in cdf.items():
        hist_equalized[key] = round(value * 255) #round off the values 
        
    return hist_equalized



def mapping(orignalEqHist, specifiedEqHist):
    
    dict1 = orignalEqHist
    dict2 = specifiedEqHist
    
    dict3 = {value_dict1: None for key_dict1, value_dict1 in dict1.items()}

    # Map values of dict3 to the nearest keys in dict2
    for value_dict1 in dict3.keys():
        closest_key_dict2 = min(dict2, key=lambda key_dict2: abs(value_dict1 - dict2[key_dict2]))
        dict3[value_dict1] = closest_key_dict2

    equalizedHist = {}  # Initialize an empty dictionary for dict4

    # Iterate through dict1 and dict3 to create dict4
    for key_dict1, value_dict1 in dict1.items():
        if value_dict1 in dict3:
            equalizedHist[key_dict1] = dict3[value_dict1]

    return equalizedHist



def map_pixel(hist_equalized,image):
    
    # Calculate the scale factor for histogram equalization
    scale_factor = 255.0 / (image.shape[0] * image.shape[1])
    
    # make a copy of orignal image
    equalized_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            equalized_image[i, j] = int(hist_equalized[image[i, j]])

    return equalized_image



def plotting1(orignalImgEqualizedHist,specImgEqualizedHist,equalizedImage,orignalImage,histSpecifiedImage):
    
    # calculating the histogram of both images
    orignal_hist = calculate_histogram(orignalImage)
    specifed_hist = calculate_histogram(histSpecifiedImage)
    equalized_hist = calculate_histogram(equalizedImage)

    orignal_hist_x = list(orignal_hist.keys())
    orignal_hist_y = list(orignal_hist.values())
    
    specified_hist_x = list(specifed_hist.keys())
    specified_hist_y = list(specifed_hist.values())
    
    orignalImgEqualizedHist_x = list(orignalImgEqualizedHist.keys())
    orignalImgEqualizedHist_y = list(orignalImgEqualizedHist.values())
    
    specImgEqualizedHist_x = list(specImgEqualizedHist.keys())
    specImgEqualizedHist_y = list(specImgEqualizedHist.values())
    
    equalized_hist_x = list(equalized_hist.keys())
    equalized_hist_y = list(equalized_hist.values())
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(8, 8))

    # Plot the first image on the left subplot
    axes[0,0].imshow(orignalImage, cmap='gray')
    axes[0,0].set_title('Orignal Image')

    axes[0,1].bar(orignal_hist_x, orignal_hist_y, width=1, align='center', color='gray')
    axes[0,1].set_title('Orignal Image Hist')

    # Plot the second image on the right subplot
    axes[1,0].imshow(histSpecifiedImage, cmap='gray')
    axes[1,0].set_title('Specified Image')

    axes[1,1].bar(specified_hist_x, specified_hist_y, width=1, align='center', color='gray')
    axes[1,1].set_title('Specified Image hist')

    axes[2,0].imshow(equalizedImage, cmap='gray')
    axes[2,0].set_title('Final Image')
    
    axes[2,1].bar(equalized_hist_x, equalized_hist_y, width=1, align='center', color='gray')
    axes[2,1].set_title('Final Image Hist')
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()



def plotting2(orignalImgEqualizedHist,specImgEqualizedHist,equalizedImage,orignalImage,histSpecifiedImage):
    # calculating the histogram of both images
    orignal_hist = calculate_histogram(orignalImage)
    specifed_hist = calculate_histogram(histSpecifiedImage)
    equalized_hist = calculate_histogram(equalizedImage)

    orignal_hist_x = list(orignal_hist.keys())
    orignal_hist_y = list(orignal_hist.values())
    
    specified_hist_x = list(specifed_hist.keys())
    specified_hist_y = list(specifed_hist.values())
    
    orignalImgEqualizedHist_x = list(orignalImgEqualizedHist.keys())
    orignalImgEqualizedHist_y = list(orignalImgEqualizedHist.values())
    
    specImgEqualizedHist_x = list(specImgEqualizedHist.keys())
    specImgEqualizedHist_y = list(specImgEqualizedHist.values())
    
    equalized_hist_x = list(equalized_hist.keys())
    equalized_hist_y = list(equalized_hist.values())
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    # Plot the first image on the left subplot
    axes[0].bar(orignal_hist_x, orignal_hist_y, width=1, align='center', color='blue')
    axes[0].set_title('Orignal Image Hist')

    # Plot the second image on the right subplot
    axes[1].bar(specified_hist_x, specified_hist_y, width=1, align='center', color='green')
    axes[1].set_title('Reference Image hist')

    axes[2].bar(equalized_hist_x, equalized_hist_y, width=1, align='center', color='red')
    axes[2].set_title('Final Image Hist')
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    



# code starts here
imagePath1 = '/home/pratik/courses/674/program/database/pgm/f_16.pgm'
orignalImage = importImage(imagePath1)

imagePath2 = '/home/pratik/courses/674/program/database/pgm/peppers.pgm'
histSpecifiedImage = importImage(imagePath2)

#calculate Histogram
orignalImgHist = calculate_histogram(orignalImage)
specifiedImgHist = calculate_histogram(histSpecifiedImage)

#calculate equalized histogram

orignalImgEqualizedHist = calculate(orignalImgHist)
specImgEqualizedHist = calculate(specifiedImgHist)

# mapping function to calculate histogram of resultant image
equalizedHist = mapping(orignalImgEqualizedHist, specImgEqualizedHist)

# mapping pixel
equalizedImage = map_pixel(equalizedHist,orignalImage)
#plotting

plotting1(orignalImgEqualizedHist,specImgEqualizedHist,equalizedImage,orignalImage,histSpecifiedImage)
plotting2(orignalImgEqualizedHist,specImgEqualizedHist,equalizedImage,orignalImage,histSpecifiedImage)



test_image = np.array([[10, 20, 30, 40, 50],
                       [40, 20, 30, 30, 10],
                       [10, 20, 30, 50, 40],
                       [10, 40, 30, 50, 20],
                       [10, 10, 40, 30, 30]], dtype=np.uint8)

specified_image = np.array([[10, 10, 20, 30, 50],
                       [10, 10, 30, 40, 10],
                       [10, 20, 30, 10, 20],
                       [30, 30, 50, 40, 40],
                       [20, 40, 30, 20, 10]], dtype=np.uint8)


#calculate Histogram
orignalImgHist = calculate_histogram(test_image)
specifiedImgHist = calculate_histogram(specified_image)

#calculate equalized histogram
orignalImgEqualizedHist = calculate(orignalImgHist)
specImgEqualizedHist = calculate(specifiedImgHist)


# mapping function to calculate histogram of resultant image
equalizedHist = mapping(orignalImgEqualizedHist, specImgEqualizedHist)

# mapping pixel
equalizedImage = map_pixel(equalizedHist,test_image)


print("orignal Image Histogram")
print(orignalImgHist)
print("\n")
print("orignal Image Equalized Histogram")
print(orignalImgEqualizedHist)
print("\n")
print("specified Image Histogram")
print(specifiedImgHist)
print("\n")
print("specified Image Equalized Histogram")
print(specImgEqualizedHist)
print("\n")
print("Equalized Image Histogram")
print(equalizedHist)




#plotting
plotting1(orignalImgEqualizedHist,specImgEqualizedHist,equalizedImage,test_image,specified_image)


# In[ ]:





# In[ ]:




