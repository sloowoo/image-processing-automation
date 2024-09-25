import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def calculate_histogram(image):
        """Calculate histogram for a grayscale image."""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Calculate histogram
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        return histogram

    @staticmethod
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

    @staticmethod
    def convert_to_gray(image):
        """Convert color image to grayscale."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def resize_and_align_images(images):
        """Resize and align a list of images."""
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)

        aligned_images = []
        for image in images:
            height_diff = max_height - image.shape[0]
            width_diff = max_width - image.shape[1]
            top = height_diff // 2
            bottom = height_diff - top
            left = width_diff // 2
            right = width_diff - left
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            aligned_images.append(padded_image)

        return aligned_images
    
    @staticmethod
    def create_output_image(equalized_image):
        return equalized_image

    @staticmethod
    def draw_histogram(output_image, histogram, hist_height, offset, start_col, end_col):
        """Draw histogram."""
        hist_width = end_col - start_col

        # Normalize the histogram
        max_val = np.max(histogram)
        normalized_hist = (histogram / max_val) * hist_height

        # Calculate bin width
        bin_width = hist_width // len(histogram)

        # Draw the histogram
        for i in range(len(histogram)):
            # Draw a rectangle for each bin
            cv2.rectangle(output_image, (start_col + bin_width * i, hist_height - int(normalized_hist[i]) + offset),
                          (start_col + bin_width * (i + 1), hist_height + offset), color=(255, 0, 0), thickness=-1)

        # Add a line bar to separate histograms
        cv2.line(output_image, (end_col, 0), (end_col, hist_height + offset), color=(0, 255, 0), thickness=3)
