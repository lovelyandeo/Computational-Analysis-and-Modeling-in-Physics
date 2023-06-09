from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
# You can edit this manually with the operations you need for the image
from skimage.morphology import erosion, dilation, closing, opening

def extract_features_graphs(img_name, lower_threshold, upper_threshold, element, operation1, operation2, properties):
    I = Image.open(img_name)
    I_gray = I.convert('L')

    # Calculating the histogram
    count, cells = I_gray.histogram(), list(range(256))
    plt.plot(cells, count)
    plt.title('Histogram of the image')
    plt.xlabel('grayscale')
    plt.ylabel('pixels')

    # Specifying the range values for the threshold
    threshold_range = (lower_threshold, upper_threshold)

    # Thresholding the image within the range
    BW = np.logical_and(np.array(I_gray) > threshold_range[0], np.array(I_gray) < threshold_range[1])
    seg = operation2(operation1(BW, element), element)
    labeled_image = label(seg)

    # Printing the original image
    plt.figure(figsize=(13, 8))
    plt.imshow(I)
    plt.title('Original Image with Numbered Objects')

    # Adding number tag on the objects of interest
    for region in regionprops(labeled_image):
        y, x = region.centroid
        plt.text(x, y, region.label, color='red', fontsize=12)
    
    # Printing the labeled image as a color bar
    plt.figure(figsize=(12, 7))
    plt.imshow(labeled_image, cmap='plasma')
    plt.colorbar()
    plt.title('Labeled Image')
    plt.show()

    # Displaying the properties or features as a table
    df = pd.DataFrame(regionprops_table(labeled_image, properties=properties))
    df = df.reset_index(drop=True)
    
    return df

def extract_features_only(img_name, lower_threshold, upper_threshold, element, operation1, operation2, properties):
    I = Image.open(img_name)
    I_gray = I.convert('L')
    I_rgb = I.convert('RGB')

    # Specifying the range values for the threshold
    threshold_range = (lower_threshold, upper_threshold)

    # Thresholding the image within the range
    BW = np.logical_and(np.array(I_gray) > threshold_range[0], np.array(I_gray) < threshold_range[1])
    seg = operation2(operation1(BW, element), element)
    labeled_image = label(seg)

    # Find the largest object (assuming it's the desired fruit)
    region_props = regionprops_table(labeled_image, properties=properties)
    largest_object_idx = np.argmax(region_props['area'])
    largest_object_label = region_props['label'][largest_object_idx]

    # Apply mask to focus on the region of the largest object
    mask = labeled_image == largest_object_label
    object_pixels = I_rgb.copy()
    object_pixels[np.logical_not(mask)] = 0

    # Compute the histogram of color values for the object pixels
    hist_values = []
    for channel in range(3):  # Iterate over RGB channels
        hist = np.histogram(object_pixels.getchannel(channel).flatten(), bins=256, range=(0, 256))[0]
        hist_values.extend(hist)

    # Display the properties or features as a table
    df = pd.DataFrame(regionprops_table(labeled_image, properties=properties))
    df['hist_values'] = hist_values  # Add histogram color values as a feature column
    df = df.reset_index(drop=True)
    
    return df

