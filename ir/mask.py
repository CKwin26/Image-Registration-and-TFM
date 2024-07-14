import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk
def main(imgpath,outp):
    def adjust_threshold(image, threshold_scale):
        """
        Adjust the threshold of the image based on the given scale.

        Args:
        image (ndarray): Input image.
        threshold_scale (float): Scale to adjust the threshold.

        Returns:
        ndarray: Thresholded binary image.
        """
        otsu_thresh = threshold_otsu(image)
        adjusted_thresh = otsu_thresh * threshold_scale
        binary_image = image < adjusted_thresh  # Invert the binary mask
        return binary_image

    # Load the image
    image_path = imgpath
    image = imageio.imread(image_path)

    # Apply Gaussian blur to reduce noise
    blurred_image = gaussian(image, sigma=2)

    # Adjust the threshold
    threshold_scale = 1.3  # Increase threshold by 30%
    adjusted_binary = adjust_threshold(blurred_image, threshold_scale)

    # Remove small objects
    adjusted_binary = remove_small_objects(adjusted_binary, min_size=1000)

    # Apply binary closing
    adjusted_binary = binary_closing(adjusted_binary, selem=disk(5))

    # Convert binary image to uint8 type for contour detection
    adjusted_binary_uint8 = (adjusted_binary * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(adjusted_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(adjusted_binary_uint8)

    # Draw only the largest contour filled on the mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)  # Fill the largest contour

    # Invert the mask colors
    inverted_mask = cv2.bitwise_not(mask)

    # Plot the result
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(adjusted_binary, cmap='gray')
    axs[0].set_title('Binary Image')
    axs[1].imshow(inverted_mask, cmap='gray')
    axs[1].set_title('Inverted Mask with Largest Contour Filled')

    # Hide axes for better visualization
    for ax in axs:
        ax.axis('off')

    # Save the plot
    output_path = outp
    plt.savefig(output_path)

    # Show the plot
    plt.show()
