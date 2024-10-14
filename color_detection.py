import cv2
import numpy as np
import pandas as pd

# Load the dataset
dataset = pd.read_csv('colors.csv')

# Define a function to detect colors
def detect_colors(image):
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the colors in the image
    colors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            colors.append((r, g, b))

    # Calculate the RGB values of the detected colors
    rgb_values = np.array(colors)

    # Determine the color name
    color_names = []
    for rgb in rgb_values:
        min_distance = float('inf')
        color_name = ''
        for index, row in dataset.iterrows():
            distance = np.linalg.norm(rgb - row[['R', 'G', 'B']])
            if distance < min_distance:
                min_distance = distance
                color_name = row['Color']
        color_names.append(color_name)

    return color_names

# Load an image
image = cv2.imread('image.jpg')

# Detect the colors in the image
color_names = detect_colors(image)

# Print the color names
print(color_names)
