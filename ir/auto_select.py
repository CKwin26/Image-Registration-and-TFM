import imageio
import pandas as pd
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects, label
from scipy.ndimage.measurements import center_of_mass
from skimage import exposure
import random

def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image, clip_limit=5.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(27,27))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def adjust_threshold(image, threshold_scale):
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_scale
    binary_image = image < adjusted_thresh  # Invert the binary mask
    return binary_image


def create_circle_mask(img, radius=1000):
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_image

def create_largest_contour_mask(image, threshold_scale=1.8):
    def adjust_threshold(image, threshold_scale):
        otsu_thresh = threshold_otsu(image)
        adjusted_thresh = otsu_thresh * threshold_scale
        binary_image = image < adjusted_thresh  # Invert the binary mask
        return binary_image

    blurred_image = gaussian(image, sigma=2)
    adjusted_binary = adjust_threshold(blurred_image, threshold_scale)
    adjusted_binary = remove_small_objects(adjusted_binary, min_size=100)
    adjusted_binary = binary_closing(adjusted_binary, selem=disk(5))
    adjusted_binary_uint8 = (adjusted_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(adjusted_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(adjusted_binary_uint8)
    if contours and len(contours) > 1:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        second_largest_contour = contours[1]
        cv2.drawContours(mask, [second_largest_contour], -1, (255), thickness=cv2.FILLED)
    elif contours:
        cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)
    return mask

def apply_mask(original_image, binary_mask):
    masked_image = np.where(binary_mask, 0, original_image)  # Set mask area to 0 (black), keep other areas unchanged
    return masked_image

def mask(img):
    xa = 400  # Modify to change the x crop area
    ya = 250  # Modify to change the y crop area
    height, width = img.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    center_x, center_y = width // 2, height // 2
    x1, y1 = center_x - (width // 2 - xa), center_y - (height // 2 - ya)
    x2, y2 = center_x + (width // 2 - xa), center_y + (height // 2 - ya)
    mask[y1:y2, x1:x2] = 255

    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return masked_image

def find_top_keypoints(image, keypoints, descriptors, mask, num_keypoints=2000):
    gradients = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if mask[y, x] != 0:  # Exclude keypoints inside the mask
            continue
        patch = image[max(0, y - 8):y + 8, max(0, x - 8):x + 8]
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            gradients.append(0)
        else:
            gradients.append(np.std(patch))

    top_indices = np.argsort(gradients)[-num_keypoints:]
    top_keypoints = [keypoints[i] for i in top_indices]
    top_descriptors = descriptors[top_indices]
    return top_keypoints, top_descriptors

def find_new_keypoints(reference_image, moving_image, keypoints, descriptors, mask, patch_size=100, hamming_threshold=130):
    orb = cv2.ORB_create()
    new_keypoints = []
    matches_list = []
    distances = []

    for kp, des in zip(keypoints, descriptors):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = moving_image[max(0, y - patch_size // 2):min(moving_image.shape[0], y + patch_size // 2), max(0, x - patch_size // 2):min(moving_image.shape[1], x + patch_size // 2)]
        kp2, des2 = orb.detectAndCompute(patch, None)

        if des2 is not None and len(kp2) > 0:
            best_kp2 = max(kp2, key=lambda k: k.response)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(np.array([des]), np.array([des2[kp2.index(best_kp2)]]))
            if matches and matches[0].distance < hamming_threshold:
                best_match = matches[0]
                best_match_pt = best_kp2.pt
                new_x = int(best_match_pt[0] + max(0, x - patch_size // 2))
                new_y = int(best_match_pt[1] + max(0, y - patch_size // 2))

                if mask[new_y, new_x] == 0:  # Include only points outside the mask
                    new_keypoints.append(cv2.KeyPoint(new_x, new_y, kp.size))
                    matches_list.append((kp, cv2.KeyPoint(new_x, new_y, kp.size)))
                    distances.append(best_match.distance)

    return new_keypoints, matches_list, distances

def compute_centroid(binary_mask):
    labeled_mask, num_labels = label(binary_mask, return_num=True)
    largest_region = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    largest_region_mask = labeled_mask == largest_region
    centroid = center_of_mass(largest_region_mask)
    return int(centroid[1]), int(centroid[0])  # Return (x, y)

def save_to_excel(matches_list, distances, base_point, output_path):
    data = []
    for i, ((kp1, kp2), dist) in enumerate(zip(matches_list, distances)):
        ref_x, ref_y = int(kp1.pt[0]), int(kp1.pt[1])
        mov_x, mov_y = int(kp2.pt[0]), int(kp2.pt[1])
        data.append({
            "Match Index": i + 1,
            "Reference X": ref_x - base_point[0],
            "Reference Y": ref_y - base_point[1],
            "Moving X": mov_x - base_point[0],
            "Moving Y": mov_y - base_point[1],
            "Distance": dist
        })

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

def plot_images_with_keypoints(reference_image, moving_image, keypoints_ref, keypoints_mov, matches_list, distances, output_dir, suffix):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot reference image and keypoints
    axs[0].imshow(reference_image, cmap='gray')
    axs[0].set_title('Reference Image with Keypoints')
    for kp in keypoints_ref:
        axs[0].plot(kp.pt[0], kp.pt[1], 'r.', markersize=5)

    # Plot moving image and keypoints
    axs[1].imshow(moving_image, cmap='gray')
    axs[1].set_title('Moving Image with Keypoints')
    for kp in keypoints_mov:
        axs[1].plot(kp.pt[0], kp.pt[1], 'r.', markersize=5)

    # Plot matching arrows
    colors = plt.cm.hsv(np.linspace(0, 1, len(matches_list))).tolist()  # Generate unique colors
    for i, ((kp1, kp2), dist) in enumerate(zip(matches_list, distances)):
        color = tuple([int(c * 255) for c in colors[i]])
        ref_x, ref_y = int(kp1.pt[0]), int(kp1.pt[1])
        mov_x, mov_y = int(kp2.pt[0]), int(kp2.pt[1])

        if dist > 20:
            axs[0].arrow(ref_x, ref_y, -mov_x + ref_x, -mov_y + ref_y, head_width=10, head_length=15, fc=colors[i], ec=colors[i])
            axs[1].arrow(mov_x, mov_y, ref_x - mov_x, ref_y - mov_y, head_width=10, head_length=15, fc=colors[i], ec=colors[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{suffix}.png'))
    plt.show()

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

def save_to_excel(matches_list, distances, output_path):
    data = []
    for i, ((kp1, kp2), dist) in enumerate(zip(matches_list, distances)):
        ref_x, ref_y = int(kp1.pt[0]), int(kp1.pt[1])
        mov_x, mov_y = int(kp2.pt[0]), int(kp2.pt[1])
        data.append({
            "Match Index": i + 1,
            "x1": ref_x,
            "y1": ref_y,
            "x2": mov_x,
            "y2": mov_y,
            "Distance": dist
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
            if i != 1 or j != 1:
                x_start = j * region_size[1]
                y_start = i * region_size[0]
                x_end = (j + 1) * region_size[1]
                y_end = (i + 1) * region_size[0]
                regions.append((x_start, y_start, x_end, y_end))

    return regions
def enhance_contrast_automatically(reference_image, moving_image, regions, outp,num_keypoints_threshold=200, max_iterations=30):
    contrast_limits = [5.0] * len(regions)  # Initial contrast limits for each region
    orb = cv2.ORB_create()

    enhanced_ref_image = reference_image.copy()
    enhanced_mov_image = moving_image.copy()

    keypoints_sections = {}
    descriptors_sections = {}

    matches_data = []  # List to store match data for each section

    max_matches = 0
    best_section_index = -1
    best_section_keypoints = None
    best_section_descriptors = None

    for i, (x_start, y_start, x_end, y_end) in enumerate(regions):
        section_key = f'section_{i+1}'
        keypoints_sections[section_key] = []
        descriptors_sections[section_key] = []

        for iteration in range(max_iterations):
            sector_ref = reference_image[y_start:y_end, x_start:x_end]
            sector_mov = moving_image[y_start:y_end, x_start:x_end]

            # Calculate the 1st percentile threshold
            p1_ref = np.percentile(sector_ref[sector_ref > 0], 5)
            p1_mov = np.percentile(sector_mov[sector_mov > 0], 5)

            # Create masks for enhancing and reducing regions
            mask_enhance_ref = sector_ref < p1_ref
            mask_reduce_ref = sector_ref >= p1_ref
            mask_enhance_mov = sector_mov < p1_mov
            mask_reduce_mov = sector_mov >= p1_mov

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=contrast_limits[i], tileGridSize=(27, 27))
            enhanced_ref_sector = clahe.apply(sector_ref)
            enhanced_mov_sector = clahe.apply(sector_mov)

            # Reduce contrast in other regions
            reduced_ref_sector = (sector_ref * 0.5).astype(np.uint8)
            reduced_mov_sector = (sector_mov * 0.5).astype(np.uint8)

            # Merge enhanced and reduced regions
            final_ref_sector = np.where(mask_enhance_ref, enhanced_ref_sector, reduced_ref_sector)
            final_mov_sector = np.where(mask_enhance_mov, enhanced_mov_sector, reduced_mov_sector)

            # Update the enhanced images with the enhanced sectors
            enhanced_ref_image[y_start:y_end, x_start:x_end] = final_ref_sector
            enhanced_mov_image[y_start:y_end, x_start:x_end] = final_mov_sector

            # Detect keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(final_ref_sector, None)
            kp2, des2 = orb.detectAndCompute(final_mov_sector, None)

            '''if des1 is None or des2 is None:
                break'''

            # Match keypoints using find_new_keypoints
            new_keypoints, matches_list, distances = find_new_keypoints(final_ref_sector, final_mov_sector, kp1, des1, np.zeros_like(mask_enhance_mov))

            num_matches = len(matches_list)
            print(f"Region {i}, iteration {iteration}: Number of matches = {num_matches}")

            # Check if the number of matches is greater than the threshold
            '''if num_matches >= num_keypoints_threshold:
                # Save match data
                max_matches=0
                for (kp_ref, kp_mov), dist in zip(matches_list, distances):
                    ref_x, ref_y = int(kp_ref.pt[0]), int(kp_ref.pt[1])
                    mov_x, mov_y = int(kp_mov.pt[0]), int(kp_mov.pt[1])
                    matches_data.append({
                        'Section': i + 1,
                        'Ref_X': ref_x + x_start,
                        'Ref_Y': ref_y + y_start,
                        'Mov_X': mov_x + x_start,
                        'Mov_Y': mov_y + y_start,
                        'Distance': dist
                    })'''

                #if kp1 is not None:
                 #   keypoints_sections[section_key].extend(kp1)
                #if des1 is not None:
                 #   descriptors_sections[section_key].extend(des1)
                #break
            max_matches +=1
            # Track the section with the highest number of matches
            if max_matches >= max_iterations:
                print('max')
                max_matches=0
                best_section_index = i
                best_section_keypoints = new_keypoints
                best_section_descriptors = descriptors_sections[section_key]
                 # Enhance contrast using CLAHE
                clahe = cv2.createCLAHE(clipLimit=contrast_limits[i], tileGridSize=(27, 27))
                enhanced_ref_sector = clahe.apply(sector_ref)
                enhanced_mov_sector = clahe.apply(sector_mov)

                # Reduce contrast in other regions
                reduced_ref_sector = (sector_ref * 0.5).astype(np.uint8)
                reduced_mov_sector = (sector_mov * 0.5).astype(np.uint8)

                # Merge enhanced and reduced regions
                final_ref_sector = np.where(mask_enhance_ref, enhanced_ref_sector, reduced_ref_sector)
                final_mov_sector = np.where(mask_enhance_mov, enhanced_mov_sector, reduced_mov_sector)

                # Update the enhanced images with the enhanced sectors
                enhanced_ref_image[y_start:y_end, x_start:x_end] = final_ref_sector
                enhanced_mov_image[y_start:y_end, x_start:x_end] = final_mov_sector

                # Detect keypoints and descriptors
                kp1, des1 = orb.detectAndCompute(final_ref_sector, None)
                kp2, des2 = orb.detectAndCompute(final_mov_sector, None)
                new_keypoints, matches_list, distances = find_new_keypoints(final_ref_sector, final_mov_sector, kp1, des1, np.zeros_like(mask_enhance_mov))

                # Save match data
                for (kp_ref, kp_mov), dist in zip(matches_list, distances):
                    ref_x, ref_y = int(kp_ref.pt[0]), int(kp_ref.pt[1])
                    mov_x, mov_y = int(kp_mov.pt[0]), int(kp_mov.pt[1])
                    matches_data.append({
                        'Section': i + 1,
                        'x1': ref_x + x_start,
                        'y1': ref_y + y_start,
                        'x2': mov_x + x_start,
                        'y2': mov_y + y_start,
                        'Distance': dist
                    })

                if kp1 is not None:
                    keypoints_sections[section_key].extend(kp1)
                if des1 is not None:
                    descriptors_sections[section_key].extend(des1)
                break
            contrast_limits[i] += 1.0  # Increase contrast limit for further enhancement

        # Use the best found contrast limit for this region
        contrast_limits[i] = contrast_limits[i]




    # Save matches data to Excel
    matches_df = pd.DataFrame(matches_data)
    output_dir = outp
    matches_df.to_excel(os.path.join(outp, 'matches_data.xlsx'), index=False)

    return enhanced_ref_image, enhanced_mov_image,matches_df



def main(refp,movp,outp):
    # Set image paths and output paths
    reference_image_path = refp
    moving_image_path = movp
    output_dir = outp

    # Create output directory (if it does not exist)
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    image_paths = [reference_image_path, moving_image_path]
    images = load_images(image_paths)
    reference_image = images[0]
    moving_image = images[1]


    # Create and apply masks
    circle_mask, masked_reference_image_circle = create_circle_mask(reference_image)
    largest_contour_mask = create_largest_contour_mask(masked_reference_image_circle)
    inverted_largest_contour_mask = cv2.bitwise_not(largest_contour_mask)
    combined_mask = cv2.bitwise_not(cv2.bitwise_and(circle_mask, inverted_largest_contour_mask))
    # Create and apply masks
    circle_mask2, masked_reference_image_circle2 = create_circle_mask(moving_image)
    largest_contour_mask2 = create_largest_contour_mask(masked_reference_image_circle2)
    inverted_largest_contour_mask2 = cv2.bitwise_not(largest_contour_mask2)
    combined_mask2 = cv2.bitwise_not(cv2.bitwise_and(circle_mask2, inverted_largest_contour_mask2))

    masked_reference_image = apply_mask(reference_image, combined_mask)
    masked_second_image = apply_mask(moving_image, combined_mask2)

    # Divide images into regions
    regions = divide_image_into_regions(masked_reference_image)

    # Automatically enhance contrast and generate keypoints and descriptors
    enhanced_ref_image, enhanced_mov_image,excel = enhance_contrast_automatically(masked_reference_image, masked_second_image, regions,outp)
    return excel

if __name__ == "__main__":
    main()
