import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk, label
from scipy.ndimage.measurements import center_of_mass

def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def adjust_threshold(image, threshold_scale):
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_scale
    binary_image = image < adjusted_thresh  # Invert the binary mask
    return binary_image

def apply_mask(original_image, binary_mask):
    masked_image = np.where(binary_mask, 0, original_image)  # Set mask area to 0 (black), keep other areas unchanged
    return masked_image

def find_top_keypoints(image, keypoints, descriptors, mask, num_keypoints=500):
    gradients = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if mask[y, x] != 0:  # Exclude keypoints inside the mask
            continue
        patch = image[max(0, y-8):y+8, max(0, x-8):x+8]
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            gradients.append(0)
        else:
            gradients.append(np.std(patch))
    
    top_indices = np.argsort(gradients)[-num_keypoints:]
    top_keypoints = [keypoints[i] for i in top_indices]
    top_descriptors = descriptors[top_indices]
    return top_keypoints, top_descriptors

def find_new_keypoints(reference_image, moving_image, keypoints, descriptors, mask, patch_size=100, hamming_threshold=180):
    orb = cv2.ORB_create()
    new_keypoints = []
    matches_list = []
    distances = []

    for kp, des in zip(keypoints, descriptors):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = moving_image[max(0, y-patch_size//2):min(moving_image.shape[0], y+patch_size//2), max(0, x-patch_size//2):min(moving_image.shape[1], x+patch_size//2)]
        kp2, des2 = orb.detectAndCompute(patch, None)
        
        if des2 is not None and len(kp2) > 0:
            # Find the keypoint with the highest response value in the patch
            best_kp2 = max(kp2, key=lambda k: k.response)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(np.array([des]), np.array([des2[kp2.index(best_kp2)]]))
            if matches and matches[0].distance < hamming_threshold:
                best_match = matches[0]
                best_match_pt = best_kp2.pt
                new_x = int(best_match_pt[0] + max(0, x-patch_size//2))
                new_y = int(best_match_pt[1] + max(0, y-patch_size//2))
                
                # Check if the new keypoint is inside the mask
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

def main(image_paths):
    output_dir = 'C:\\Users\\austa\\Downloads\\0.4'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = load_images(image_paths)
    reference_image = cv2.convertScaleAbs(images[0])
    moving_image = cv2.convertScaleAbs(images[1])

    # Apply Gaussian blur and threshold to create binary masks for both images
    blurred_reference = gaussian(reference_image, sigma=2)
    blurred_moving = gaussian(moving_image, sigma=2)

    threshold_scale = 1.3  # Increase threshold by 30%

    adjusted_binary_ref = adjust_threshold(blurred_reference, threshold_scale)
    adjusted_binary_ref = remove_small_objects(adjusted_binary_ref, min_size=1000)
    adjusted_binary_ref = binary_closing(adjusted_binary_ref, selem=disk(5))

    adjusted_binary_mov = adjust_threshold(blurred_moving, threshold_scale)
    adjusted_binary_mov = remove_small_objects(adjusted_binary_mov, min_size=1000)
    adjusted_binary_mov = binary_closing(adjusted_binary_mov, selem=disk(5))

    # Show and save the mask for verification
    plt.imshow(adjusted_binary_ref, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'binary_mask.png'))
    plt.show()

    # Mask the reference and moving images
    masked_reference_image = apply_mask(reference_image, adjusted_binary_ref)
    masked_moving_image = apply_mask(moving_image, adjusted_binary_mov)

    # Enhance contrast
    enhanced_reference_image = enhance_contrast(masked_reference_image)
    enhanced_moving_image = enhance_contrast(masked_moving_image)

    # Compute centroid of the reference image mask
    base_point = compute_centroid(adjusted_binary_ref)
    print(f"Centroid of the mask (base point): {base_point}")

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(enhanced_reference_image, None)

    # Filter top keypoints using the reference image mask
    valid_keypoints = []
    valid_descriptors = []

    for kp, des in zip(kp1, des1):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if adjusted_binary_ref[y, x] == 0:  # Include points outside the tumor mask
            valid_keypoints.append(kp)
            valid_descriptors.append(des)

    valid_descriptors = np.array(valid_descriptors)

    top_keypoints, top_descriptors = find_top_keypoints(enhanced_reference_image, valid_keypoints, valid_descriptors, adjusted_binary_ref)

    # Find new keypoints using the moving image mask
    new_keypoints, matches_list, distances = find_new_keypoints(enhanced_reference_image, enhanced_moving_image, top_keypoints, top_descriptors, adjusted_binary_mov)
    
    img1_with_kp = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    img2_with_kp = cv2.cvtColor(moving_image, cv2.COLOR_GRAY2BGR)

    colors = plt.cm.hsv(np.linspace(0, 1, len(matches_list))).tolist()  # Generate unique colors

    for i, (kp1, kp2) in enumerate(matches_list):
        color = tuple([int(c * 255) for c in colors[i]])
        cv2.circle(img1_with_kp, (int(kp1.pt[0]), int(kp1.pt[1])), 5, color, -1)
        cv2.circle(img2_with_kp, (int(kp2.pt[0]), int(kp2.pt[1])), 5, color, -1)
        # Draw arrowed line from centroid to keypoints
        cv2.arrowedLine(img1_with_kp, base_point, (int(kp1.pt[0]), int(kp1.pt[1])), color, 2, tipLength=0.1)
        cv2.arrowedLine(img2_with_kp, base_point, (int(kp2.pt[0]), int(kp2.pt[1])), color, 2, tipLength=0.1)

    imageio.imwrite(os.path.join(output_dir, 'reference_image_with_kp.tif'), img1_with_kp)
    imageio.imwrite(os.path.join(output_dir, 'moving_image_with_new_kp.tif'), img2_with_kp)

    print("Key points in reference image saved.")
    print("New key points in moving image saved.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img1_with_kp, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Reference Image with Keypoints')
    axs[1].imshow(cv2.cvtColor(img2_with_kp, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Moving Image with New Keypoints')
    plt.show()

    # Save matched points and distances to Excel
    excel_output_path = os.path.join(output_dir, 'matched_points_and_distances.xlsx')
    save_to_excel(matches_list, distances, base_point, excel_output_path)
    print(f"Matched points and distances saved to {excel_output_path}")

if __name__ == "__main__":
    image_paths = [
        'C:\\Users\\austa\\Downloads\\0.4\\reference_image_cropped.tif',
        'C:\\Users\\austa\\Downloads\\0.4\\aligned_image_cropped.tif'
    ]
    main(image_paths)
