import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random





'''
START OF SEGMENT 1

MIT License

Copyright (c) 2024 Enoch Kwateh Dongbo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def calculate_histogram(image):
        """Calculate histogram for a grayscale image."""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Calculate histogram
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        return histogram

def histogram_equalization(image):
        """Perform histogram equalization on a grayscale image."""
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Convert back to BGR format
        equalized_bgr_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

        return equalized_bgr_image

'''
END OF SEGMENT 1
'''

def save_equalized_image(input_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    
    # Perform histogram equalization
    equalized_image = histogram_equalization(image)
    
    # Get the output image path
    output_image_path = os.path.splitext(input_image_path)[0] + '_equalized.png'
    
    # Save the equalized image
    cv2.imwrite(output_image_path, equalized_image)
    
    print(f"Equalized image saved as: {output_image_path}")



def binarize_image(image_path, threshold):
    # Read the image in grayscale
    image = cv2.imread(image_path, 0)
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_image

def save_binarized_image(input_image_path, threshold):
    # Perform binarization
    binarized_image = binarize_image(input_image_path, threshold)
    
    # Get the output image path
    output_image_path = os.path.splitext(input_image_path)[0] + '_binarized.png'
    
    # Save the image
    cv2.imwrite(output_image_path, binarized_image)
    
    print(f"Binarized image saved as: {output_image_path}")

# Input image path
input_image_path = 'test1.jpg'
equilized_image_path = 'test1_equalized.png'

# Call the function to save the equalized image
save_equalized_image(input_image_path)
save_binarized_image(equilized_image_path, 120)


