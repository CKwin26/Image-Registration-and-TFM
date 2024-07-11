import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image):
    # Enhance contrast to capture characteristics
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def mask_center(image, mask_size):
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    half_mask_size = mask_size // 2
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[center_y - half_mask_size:center_y + half_mask_size, center_x - half_mask_size:center_x + half_mask_size] = 0
    return cv2.bitwise_and(image, mask)

def align_images_orb(reference_image, moving_image, mask_size):
    # Create ORB feature detector
    orb = cv2.ORB_create()

    # Enhance contrast
    reference_image_enhanced = enhance_contrast(reference_image)
    moving_image_enhanced = enhance_contrast(moving_image)

    # Apply mask to the images
    reference_image_masked = mask_center(reference_image_enhanced, mask_size)
    moving_image_masked = mask_center(moving_image_enhanced, mask_size)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(reference_image_masked, None)
    keypoints2, descriptors2 = orb.detectAndCompute(moving_image_masked, None)

    # Create BFMatcher for descriptor matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched point positions
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Compute affine transform matrix
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)

    # Apply affine transform to align images
    aligned_image = cv2.warpAffine(moving_image, M, (reference_image.shape[1], reference_image.shape[0]))

    return aligned_image, reference_image

def crop_image(image, x_start, x_end, y_start, y_end):
    return image[y_start:y_end, x_start:x_end]

def main(image_paths):
    output_dir = 'C:\\Users\\austa\\Downloads\\0.4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = load_images(image_paths)
    reference_image = cv2.convertScaleAbs(images[0])
    moving_image = cv2.convertScaleAbs(images[1])

    # Define mask size
    mask_size = 1000

    # Align images
    aligned_image, reference_image_masked = align_images_orb(reference_image, moving_image, mask_size)

    # Crop images to specified dimensions
    x_start, x_end = 10, 2400
    y_start, y_end = 10, 2000
    reference_image_cropped = crop_image(reference_image, x_start, x_end, y_start, y_end)
    aligned_image_cropped = crop_image(aligned_image, x_start, x_end, y_start, y_end)

    # Save images
    imageio.imwrite(os.path.join(output_dir, 'aligned_image_cropped.tif'), aligned_image_cropped)
    imageio.imwrite(os.path.join(output_dir, 'reference_image_cropped.tif'), reference_image_cropped)

    print("Aligned and cropped image saved.")
    print("Cropped reference image saved.")

    # Display aligned and cropped image and masked reference image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(reference_image_cropped, cmap='gray')
    axs[0].set_title('Cropped Reference Image')
    axs[1].imshow(aligned_image_cropped, cmap='gray')
    axs[1].set_title('Aligned and Cropped Image')
    plt.show()

if __name__ == "__main__":
    image_paths = [
        'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 0 + 88bit.tif',
        'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 4 + 88bit.tif'
    ]
    main(image_paths)
