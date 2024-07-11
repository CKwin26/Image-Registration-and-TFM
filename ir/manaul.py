import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops
from scipy.ndimage.measurements import center_of_mass

def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image, clip_limit=5.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(27, 27))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def adjust_threshold(image, threshold_scale):
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_scale
    binary_image = image < adjusted_thresh  # Invert the binary mask
    return binary_image

def find_regions(image, threshold_scale):
    blurred_image = gaussian(image, sigma=2)
    binary_image = adjust_threshold(blurred_image, threshold_scale)
    binary_image = binary_closing(binary_image, selem=disk(5))
    labeled_image, num_labels = label(binary_image, return_num=True)
    return labeled_image, num_labels, binary_image

def remove_large_objects(binary_image, max_size):
    labeled_image, num_labels = label(binary_image, return_num=True)
    sizes = np.bincount(labeled_image.ravel())
    mask_sizes = sizes < max_size
    mask_sizes[0] = 0  # background should not be removed
    binary_image = mask_sizes[labeled_image]
    binary_image = binary_closing(binary_image, selem=disk(5))  # Re-label image after removing large objects
    return binary_image

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def match_contours(reference_contours, moving_contours):
    matched_contours = []
    for ref_cnt in reference_contours:
        M = cv2.moments(ref_cnt)
        if M['m00'] == 0:
            continue
        ref_cx = int(M['m10'] / M['m00'])
        ref_cy = int(M['m01'] / M['m00'])
        ref_area = cv2.contourArea(ref_cnt)
        
        best_match = None
        min_area_diff = float('inf')
        for mov_cnt in moving_contours:
            M = cv2.moments(mov_cnt)
            if M['m00'] == 0:
                continue
            mov_cx = int(M['m10'] / M['m00'])
            mov_cy = int(M['m01'] / M['m00'])
            mov_area = cv2.contourArea(mov_cnt)
            
            if abs(mov_cx - ref_cx) <= 50 and abs(mov_cy - ref_cy) <= 50:
                area_diff = abs(mov_area - ref_area)
                if area_diff < min_area_diff and area_diff <= 0.2 * ref_area:
                    best_match = (ref_cnt, mov_cnt, (mov_cx, mov_cy, mov_area))
                    min_area_diff = area_diff
        
        if best_match:
            matched_contours.append(best_match)
    
    return matched_contours

def save_to_excel(matched_contours, output_path):
    data = []
    for i, (ref_cnt, mov_cnt, loc) in enumerate(matched_contours):
        x, y, w, h = cv2.boundingRect(ref_cnt)
        mov_x, mov_y, mov_area = loc
        data.append({
            "Match Index": i + 1,
            "Reference X": x,
            "Reference Y": y,
            "Reference Width": w,
            "Reference Height": h,
            "Reference Area": cv2.contourArea(ref_cnt),
            "Moving X": mov_x,
            "Moving Y": mov_y,
            "Moving Area": mov_area
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

def plot_images(reference_image, binary_ref, labeled_reference, moving_image, binary_mov, labeled_moving, output_dir, suffix):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(reference_image, cmap='gray')
    axs[0, 0].set_title('Reference Image')

    axs[0, 1].imshow(binary_ref, cmap='gray')
    axs[0, 1].set_title(f'Reference Image Binary ({suffix})')

    axs[0, 2].imshow(labeled_reference, cmap='nipy_spectral')
    axs[0, 2].set_title(f'Reference Image Labeled ({suffix})')

    axs[1, 0].imshow(moving_image, cmap='gray')
    axs[1, 0].set_title('Moving Image')

    axs[1, 1].imshow(binary_mov, cmap='gray')
    axs[1, 1].set_title(f'Moving Image Binary ({suffix})')

    axs[1, 2].imshow(labeled_moving, cmap='nipy_spectral')
    axs[1, 2].set_title(f'Moving Image Labeled ({suffix})')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{suffix}.png'))
    plt.show()

def divide_image_into_regions(image):
    h, w = image.shape
    regions = []
    region_size = (h // 3, w // 3)

    for i in range(3):
        for j in range(3):
            x_start = j * region_size[1]
            y_start = i * region_size[0]
            x_end = (j + 1) * region_size[1]
            y_end = (i + 1) * region_size[0]
            regions.append((x_start, y_start, x_end, y_end))
    
    return regions

def enhance_contrast_automatically(reference_image, moving_image, regions):
    clip_limits = [0.0] * len(regions)  # Initial clip limits for each region
    min_regions = 50  # Minimum number of regions required

    enhanced_ref_image = reference_image.copy()
    enhanced_mov_image = moving_image.copy()

    for i, (x_start, y_start, x_end, y_end) in enumerate(regions):
        num_labels_ref_array = []
        num_labels_mov_array = []
        while True:
            sector_ref = reference_image[y_start:y_end, x_start:x_end]
            sector_mov = moving_image[y_start:y_end, x_start:x_end]

            enhanced_ref_sector = enhance_contrast(sector_ref, clip_limits[i])
            enhanced_mov_sector = enhance_contrast(sector_mov, clip_limits[i])

            # Update the enhanced images with the enhanced sectors
            enhanced_ref_image[y_start:y_end, x_start:x_end] = enhanced_ref_sector
            enhanced_mov_image[y_start:y_end, x_start:x_end] = enhanced_mov_sector

            _, num_labels_ref, _ = find_regions(enhanced_ref_image, threshold_scale=1.3)
            _, num_labels_mov, _ = find_regions(enhanced_mov_image, threshold_scale=1.3)

            num_labels_ref_array.append(num_labels_ref)
            num_labels_mov_array.append(num_labels_mov)

            print(f"Region {i}: num_labels_ref = {num_labels_ref}, num_labels_mov = {num_labels_mov}")

            if len(num_labels_ref_array) >= 10 and len(num_labels_mov_array) >= 10:
                diffs = [abs(nr - nm) for nr in num_labels_ref_array for nm in num_labels_mov_array]
                n1=0
                n2=0
                for nr in num_labels_ref_array:
                    n1+=2
                    for nm in num_labels_mov_array:
                        n2+=2
                        if abs(nr-nm)<min_regions:
                            print('suc')
                            sector_ref = reference_image[y_start:y_end, x_start:x_end]
                            sector_mov = moving_image[y_start:y_end, x_start:x_end]

                            enhanced_ref_sector = enhance_contrast(sector_ref, n1)
                            enhanced_mov_sector = enhance_contrast(sector_mov, n2)

                            # Update the enhanced images with the enhanced sectors
                            enhanced_ref_image[y_start:y_end, x_start:x_end] = enhanced_ref_sector
                            enhanced_mov_image[y_start:y_end, x_start:x_end] = enhanced_mov_sector

                            break
                break
               
            
                
            if clip_limits[i] <= 30:
                clip_limits[i] += 2  # Increase contrast for sectors with fewer matches
                if abs(num_labels_ref-num_labels_mov)<min_regions:
                    break
    # Plot the enhanced images
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(reference_image, cmap='gray')
    axs[0, 0].set_title('Reference Image')
    axs[0, 1].imshow(enhanced_ref_image, cmap='gray')
    axs[0, 1].set_title('Enhanced Reference Image')
    axs[0, 2].imshow(moving_image, cmap='gray')
    axs[0, 2].set_title('Moving Image')
    axs[1, 0].imshow(enhanced_mov_image, cmap='gray')
    axs[1, 0].set_title('Enhanced Moving Image')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return enhanced_ref_image, enhanced_mov_image

def ensure_minimum_regions(reference_image, moving_image, regions, threshold_scale=2, min_regions=30):
    enhanced_ref_image = reference_image.copy()
    enhanced_mov_image = moving_image.copy()

    for x_start, y_start, x_end, y_end in regions:
        while True:
            sector_ref = reference_image[y_start:y_end, x_start:x_end]
            sector_mov = moving_image[y_start:y_end, x_start:x_end]
            
            labeled_ref, num_labels_ref, binary_ref = find_regions(sector_ref, threshold_scale)
            labeled_mov, num_labels_mov, binary_mov = find_regions(sector_mov, threshold_scale)

            print(num_labels_ref)
            print('nf')
            print(num_labels_mov)
            print('nm')

            if num_labels_ref >= min_regions and num_labels_mov >= min_regions:
                enhanced_ref_image[y_start:y_end, x_start:x_end] = sector_ref
                enhanced_mov_image[y_start:y_end, x_start:x_end] = sector_mov
                print('suc')
                break

            threshold_scale -= 0.05  # Adjust the threshold scale to find more regions
        print(f"Processed region: ({x_start}, {y_start}) to ({x_end}, {y_end})")
    return enhanced_ref_image, enhanced_mov_image
        

def create_tumor_mask(binary_image):
    labeled_image = label(binary_image.astype(np.uint8))
    properties = regionprops(labeled_image)
    largest_component = max(properties, key=lambda x: x.area)
    tumor_mask = labeled_image == largest_component.label
    return tumor_mask

def create_outer_mask(image_shape, margin):
    mask = np.zeros(image_shape, dtype=bool)
    mask[:margin, :] = True
    mask[-margin:, :] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True
    return mask

def align_images_orb(reference_image, moving_image, mask_size=0):
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(reference_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(moving_image, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography
    height, width = reference_image.shape
    aligned_image = cv2.warpPerspective(moving_image, h, (width, height))

    # Create masks for the aligned images
    mask_ref = np.ones(reference_image.shape, dtype=np.uint8) * 255
    mask_mov = np.ones(moving_image.shape, dtype=np.uint8) * 255

    reference_image_masked = cv2.bitwise_and(reference_image, reference_image, mask=mask_ref)
    moving_image_masked = cv2.bitwise_and(aligned_image, aligned_image, mask=mask_mov)

    return aligned_image, reference_image_masked

def main(image_paths):
    output_dir = 'C:\\Users\\austa\\Downloads\\0.4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = load_images(image_paths)
    reference_image = cv2.convertScaleAbs(images[0])
    moving_image = cv2.convertScaleAbs(images[1])
    print('prepare')
    # Divide image into regions
    regions = divide_image_into_regions(reference_image)
    print('divide')
    # Automatically enhance contrast in each region
    enhanced_ref_image, enhanced_mov_image = enhance_contrast_automatically(reference_image, moving_image, regions)
    print('enhance')
    threshold_scale_ref = 1.3
    threshold_scale_mov = 1.3
    
    # Initial binary results
    labeled_reference, num_labels_ref, binary_ref = find_regions(enhanced_ref_image, threshold_scale_ref)
    labeled_moving, num_labels_mov, binary_mov = find_regions(enhanced_mov_image, threshold_scale_mov)
    
    plot_images(enhanced_ref_image, binary_ref, labeled_reference, enhanced_mov_image, binary_mov, labeled_moving, output_dir, 'initial')

    # Ensure minimum regions in each sector
    enhanced_ref_image, enhanced_mov_image = ensure_minimum_regions(enhanced_ref_image, enhanced_mov_image, regions)
    
    labeled_reference, num_labels_ref, binary_ref = find_regions(enhanced_ref_image, threshold_scale_ref)
    labeled_moving, num_labels_mov, binary_mov = find_regions(enhanced_mov_image, threshold_scale_mov)
    
    plot_images(enhanced_ref_image, binary_ref, labeled_reference, enhanced_mov_image, binary_mov, labeled_moving, output_dir, 'adjusted')

    # Create masks
    tumor_mask_ref = create_tumor_mask(binary_ref)
    outer_mask_ref = create_outer_mask(reference_image.shape, margin=250)
    combined_mask_ref = np.logical_or(tumor_mask_ref, outer_mask_ref)

    tumor_mask_mov = create_tumor_mask(binary_mov)
    outer_mask_mov = create_outer_mask(enhanced_mov_image.shape, margin=250)
    combined_mask_mov = np.logical_or(tumor_mask_mov, outer_mask_mov)

    # Apply masks
    reference_image_masked = np.where(combined_mask_ref, 0, enhanced_ref_image)
    moving_image_masked = np.where(combined_mask_mov, 0, enhanced_mov_image)

    # Remove large objects
    binary_ref_no_large = remove_large_objects(reference_image_masked, 500)
    binary_mov_no_large = remove_large_objects(moving_image_masked, 500)

    labeled_reference_no_large, num_labels_ref_no_large = label(binary_ref_no_large, return_num=True)
    labeled_moving_no_large, num_labels_mov_no_large = label(binary_mov_no_large, return_num=True)

    # Find contours in the binary images
    reference_contours = find_contours(binary_ref_no_large)
    moving_contours = find_contours(binary_mov_no_large)

    # Match contours
    matched_contours = match_contours(reference_contours, moving_contours)

    # Draw matched contours
    ref_image_with_contours = reference_image_masked.copy()
    mov_image_with_contours = moving_image_masked.copy()
    for ref_cnt, mov_cnt, loc in matched_contours:
        x, y, w, h = cv2.boundingRect(ref_cnt)
        cv2.rectangle(ref_image_with_contours, (x, y), (x + w, y + h), (255, 255, 0), 10)
        mov_x, mov_y, _ = loc
        cv2.rectangle(mov_image_with_contours, (mov_x, mov_y), (mov_x + w, mov_y + h), (255, 255, 0), 10)

    imageio.imwrite(os.path.join(output_dir, 'reference_image_with_regions.tif'), ref_image_with_contours)
    imageio.imwrite(os.path.join(output_dir, 'aligned_image_with_regions.tif'), mov_image_with_contours)

    excel_output_path = os.path.join(output_dir, 'matched_regions.xlsx')
    save_to_excel(matched_contours, excel_output_path)
    print(f"Matched regions saved to {excel_output_path}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(ref_image_with_contours, cmap='gray')
    axs[0].set_title('Reference Image with Regions')
    axs[1].imshow(mov_image_with_contours, cmap='gray')
    axs[1].set_title('Aligned Image with Regions')
    plt.show()

if __name__ == "__main__":
    image_paths = [
        'C:\\Users\\austa\\Downloads\\0.4\\aligned_image_cropped.tif',
        'C:\\Users\\austa\\Downloads\\0.4\\reference_image_cropped.tif'
    ]
    main(image_paths)
