import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, disk

def load_images(paths):
    """Load images from the given paths."""
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image):
    """Enhance the contrast of the image using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def adjust_threshold(image, threshold_scale):
    """Adjust the threshold of the image based on Otsu's method."""
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_scale
    binary_image = image < adjusted_thresh  # Invert the binary mask
    return binary_image

def apply_mask(original_image, binary_mask):
    """Apply the binary mask to the original image."""
    masked_image = np.where(binary_mask, 0, original_image)  # Set mask area to 0 (black), keep other areas unchanged
    return masked_image

def find_min_grayvalue_point(image, x, y, box_size=10):
    """Find the point with the minimum gray value within a box around (x, y)."""
    x_min = max(0, x - box_size // 2)
    x_max = min(image.shape[1], x + box_size // 2 + 1)
    y_min = max(0, y - box_size // 2)
    y_max = min(image.shape[0], y + box_size // 2 + 1)

    min_val = np.min(image[y_min:y_max, x_min:x_max])
    min_loc = np.where(image[y_min:y_max, x_min:x_max] == min_val)
    min_x = x_min + min_loc[1][0]
    min_y = y_min + min_loc[0][0]
    
    return (min_x, min_y)

def manual_select_keypoints(images, titles=["Reference Image", "Moving Image"]):
    """Manually select keypoints on the images."""
    keypoints = [[], []]
    selected_point = [-1, -1]

    def select_point(event, x, y, flags, param):
        img_index = param["img_index"]
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find the point with the minimum gray value within a 10x10 box
            min_point = find_min_grayvalue_point(images[img_index], x, y)
            print(img_index)
            keypoints[img_index].append(min_point)
            print(keypoints[img_index])
            cv2.circle(images[img_index], min_point, 5, (0, 255, 0), -1)
            cv2.putText(images[img_index], str(len(keypoints[img_index])), min_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(titles[img_index], images[img_index])
            if img_index == 0:
                selected_point[0] = min_point
                
                draw_rectangle_on_other_image(*min_point)
            elif img_index == 1:
                selected_point[1] = min_point
                clear_rectangle_on_other_image()
               
#remove the pair
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Check if a point is selected for removal
            for i, (px, py) in enumerate(keypoints[img_index]):
                if (px - 10 <= x <= px + 10) and (py - 10 <= y <= py + 10):
                    keypoints[img_index].pop(i)
                    if len(keypoints[1 - img_index]) > i:
                        keypoints[1 - img_index].pop(i)
                    redraw_images()
                    return

    def draw_rectangle_on_other_image(x, y):
        x_min = max(0, x - 10)
        x_max = min(images[1].shape[1], x + 10)
        y_min = max(0, y - 10)
        y_max = min(images[1].shape[0], y + 10)
        cv2.rectangle(images[1], (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imshow(titles[1], images[1])

    def clear_rectangle_on_other_image():
        images[1] = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
        redraw_images()

    def redraw_images():
        for img_index in range(2):
            images[img_index] = cv2.imread(image_paths[img_index], cv2.IMREAD_GRAYSCALE)
            for i, (x, y) in enumerate(keypoints[img_index]):
                cv2.circle(images[img_index], (x, y), 5, (0, 255, 0), -1)
                cv2.putText(images[img_index], str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(titles[img_index], images[img_index])

    for i, img in enumerate(images):
        cv2.imshow(titles[i], img)
        cv2.setMouseCallback(titles[i], select_point, param={"img_index": i})
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(keypoints[0])
    print(keypoints[1])
    return keypoints[0], keypoints[1]

def compute_distances(keypoints1, keypoints2):
    """Compute distances between matched keypoints."""
    distances = []
    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(dist)
    return distances

def save_to_excel(keypoints1, keypoints2, distances, output_path):
    """Save the matched keypoints and distances to an Excel file."""
    data = []
    for i, ((x1, y1), (x2, y2), dist) in enumerate(zip(keypoints1, keypoints2, distances)):
        data.append({
            "Match Index": i + 1,
            "Reference X": x1,
            "Reference Y": y1,
            "Moving X": x2,
            "Moving Y": y2,
            "Distance": dist
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

def main(image_paths,outd):
    """Main function to process images and manually select keypoints."""
    output_dir = outd
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
    adjusted_binary_mov = binary_closing(adjusted_binary_mov, selem=(disk(5)))

    # Mask the reference and moving images
    masked_reference_image = apply_mask(reference_image, adjusted_binary_ref.copy())
    masked_moving_image = apply_mask(moving_image, adjusted_binary_mov.copy())

    # Enhance contrast
    enhanced_reference_image = enhance_contrast(masked_reference_image)
    enhanced_moving_image = enhance_contrast(masked_moving_image)

    # Manual selection of keypoints
    ref_image_for_selection = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    mov_image_for_selection = cv2.cvtColor(moving_image, cv2.COLOR_GRAY2BGR)

    print("Select keypoints on the reference and moving images")
    keypoints_ref, keypoints_mov = manual_select_keypoints([ref_image_for_selection.copy(), mov_image_for_selection.copy()], titles=["Reference Image", "Moving Image"])

    # Ensure the number of selected keypoints matches
    if len(keypoints_ref) != len(keypoints_mov):
        print(len(keypoints_ref))
        print(len(keypoints_mov))
        print(keypoints_ref)
        print(keypoints_mov)
        print("Error: The number of keypoints selected on both images must be the same.")
        return

    # Compute distances between matched keypoints
    distances = compute_distances(keypoints_ref, keypoints_mov)
    
    # Save matched points and distances to Excel
    excel_output_path = os.path.join(output_dir, 'matched_points_and_distances_manual.xlsx')
    save_to_excel(keypoints_ref, keypoints_mov, distances, excel_output_path)
    print(f"Matched points and distances saved to {excel_output_path}")

    # Draw matched keypoints on images
    for (x1, y1), (x2, y2) in zip(keypoints_ref, keypoints_mov):
        cv2.circle(ref_image_for_selection, (x1, y1), 5, (0, 255, 0), -1)
        cv2.circle(mov_image_for_selection, (x2, y2), 5, (0, 255, 0), -1)

    # Save images with keypoints
    imageio.imwrite(os.path.join(output_dir, 'reference_image_with_manual_kp.tif'), ref_image_for_selection)
    imageio.imwrite(os.path.join(output_dir, 'moving_image_with_manual_kp.tif'), mov_image_for_selection)

    print("Images with manually selected keypoints saved.")

    # Display images with keypoints
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(ref_image_for_selection, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Reference Image with Keypoints')
    axs[1].imshow(cv2.cvtColor(mov_image_for_selection, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Moving Image with Keypoints')
    plt.show()

if __name__ == "__main__":
    image_paths = [
        'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 0 + 88bit.tif',
        'C:\\Users\\austa\\Downloads\\eighth_0.4\\Day 4 + 88bit.tif'
    ]
    main(image_paths)
