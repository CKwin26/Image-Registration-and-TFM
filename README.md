# Image-Registration-and-TFM

manual select point is for select points by hand
ORB is for image registration for the image with crop and transtion and rotation
select point with ORB is for select keypoints with ORB method(mainly Hamming distance and FAST) to find the displacement of points based on the centroid of tumor

# ALIMAGE
The Alimage script is designed for image alignment and processing, utilizing ORB (Oriented FAST and Rotated BRIEF) feature matching.

It loads two input images, enhances their contrast through CLAHE (Contrast Limited Adaptive Histogram Equalization), and applies a mask to exclude the central region for feature detection.

Enhances Contrast:
    
    def enhance_contrast(image)
    
    Enhance the contrast of a grayscale image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhancing contrast works by adjusting the grayscale values of an image, increasing the difference between light and dark areas.
    This process makes details more visible by stretching or redistributing the pixel intensities.
    For example, in medical images, enhancing contrast helps distinguish subtle differences between tissues, making structures like tumors stand out more clearly for further analysis.
    
![image](https://github.com/user-attachments/assets/6762880b-36cb-4039-b988-4f7e517d6c8a)
![image](https://github.com/user-attachments/assets/e002fd09-6362-4b97-8ca9-96205f145dd1)

Applies Mask:

    def mask_center(image, mask_size)

    Apply a central mask to an image. The mask blocks out a square area in the center of the image.
    Apply mask at center to exclude the image's central region from feature detection and processing.
    This is useful when the center contains irrelevant or distracting information, such as noise or a static object that doesn't contribute to the alignment or analysis
    
![image](https://github.com/user-attachments/assets/e4dec117-1417-43b3-89a8-2602f8d34762)

By matching keypoints between the two images, it calculates an affine transformation matrix to align the moving image with the reference.
![image](https://github.com/user-attachments/assets/da375219-d8e0-45e4-aaed-f7954a57ba43)

The aligned and reference images are then cropped to a specified region and saved in an output directory. The aligned and cropped images are also displayed using Matplotlib. This tool is ideal for image registration and region-based cropping applications.

Before Enhance:
![image](https://github.com/user-attachments/assets/34657efd-3663-4b2c-8c2b-fc6c6715e7ea)
After Enhance:
![image](https://github.com/user-attachments/assets/519008ba-8df9-448d-89ca-827739c3abe4)
Maks:
![image](https://github.com/user-attachments/assets/9f0b2617-a6c6-4b18-ba63-26e7c1b272c6)


    def align_images_orb(reference_image, moving_image, mask_size)
    Align two images using ORB (Oriented FAST and Rotated BRIEF) feature detection and matching.
    This detects and matches keypoints between the two images, computes an affine transformation matrix, and applies this transformation to align the moving image with the reference.
    This process is essential for image registration tasks, ensuring that corresponding features in both images are properly aligned for accurate comparison or analysis.
    
Oriented FAST: This variant of the FAST algorithm enhances its performance by incorporating orientation information, allowing it to detect keypoints that are invariant to rotation.

Rotated BRIEF: This technique extends the BRIEF descriptor to handle rotations of the image, ensuring that the keypoint descriptors remain consistent even if the image is rotated.


# m_select

The m_select project is designed for image analysis, specifically focusing on the detection and comparison of tumor contours in medical images.

This code allows users to load two images, typically representing different time points or conditions of the same subject, and perform various operations including corp image, contrast enhancement, mask generation, and keypoint selection.

    def resize_image(image, scale=0.5, width=None, height=None):
    
    Resizes the given image based on either scale, width, or height.
    If scale is provided, it will scale the image by that factor (default 0.5).
    If width and height are provided, it will resize the image to those dimensions.
    If only width or height is provided, it will maintain the image aspect ratio.
    Raises a ValueError if neither scale, width, nor height is specified.
    
    
    adjust_threshold(image, threshold_scale):
    
    Adjusts the threshold of the input image using Otsu's method, scaled by the provided factor.
    It allows for adaptive thresholding, where the threshold level can be scaled to improve the
    distinction between different image regions. By modifying the threshold, it helps in better
    isolating relevant features (e.g., tumors) from noise, facilitating more accurate analysis and processing in subsequent steps.


    def apply_mask(original_image, binary_mask):
    
    Applies a binary mask to the input image, setting masked areas to black (0).
    This can effectively filtering out irrelevant parts, allowing for more focused feature extraction or image transformation.


Keypoint Selection:
![image](https://github.com/user-attachments/assets/6225e46c-7fbb-4c74-92bc-d7bdb6ff2645)
![image](https://github.com/user-attachments/assets/562e5142-5d78-4906-ab8a-c27a04b71166)

The tool enables users to manually select keypoints on the tumor contours, calculate distances between corresponding points, and analyze the radial displacement within defined sectors around the tumor.

    def manual_select_sector_points(image, num_sectors=6):
    
    Allows the user to manually select sector points on an image by clicking, and stores the coordinates.
    By manually selecting points, users can ensure that the sectors align with important features in the image,
    improving the accuracy of subsequent analyses.


    manual_select_keypoints(images, center, sector_points, titles=["Reference Image", "Moving Image"],scale=0.5):
    
    Allows the user to manually select keypoints on two images, store those keypoints, and associate each with a sector.
    This function enables users to manually select keypoints in two images (often a reference and a moving image) based on visual inspection,
    aiding in image registration or feature tracking tasks.
    (The selected data will be saved in excel sheet)
    
Contours:
![image](https://github.com/user-attachments/assets/90348024-2cec-4af4-9b0e-360b16342441)

The results, including matched keypoints and distance metrics, can be saved for further analysis in an Excel format. This project is intended for researchers and medical professionals interested in quantitative analysis of tumor changes over time or in response to treatment.
