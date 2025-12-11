import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image conditions as tuples.
image_conditions = [
    # ("RZ180.0", "RX-120.0"),
    # ("RZ0.0", "RX-120.0"),
    ("RZ180.0", "RX180.0"),
    ("RZ0.0", "RX180.0"),
]
def normalize_filename(filename):
    pattern = r'^([ND]-\d+-\d+)_.*?(cam.*)$'
    match = re.match(pattern, filename)
    return match.group(1) + "_" + match.group(2) if match else filename

def compute_iou_scores_with_filenames(folder1, folder2, image_conditions, replace_substring=None):
    folder2_files = {normalize_filename(f): f for f in os.listdir(folder2)}
    iou_scores, filenames = [], []

    for filename in os.listdir(folder1):
        norm_filename = normalize_filename(filename)
        if not any(cond[0] in norm_filename and cond[1] in norm_filename for cond in image_conditions):
            continue

        if norm_filename in folder2_files:
            path1 = os.path.join(folder1, filename)
            path2_name = folder2_files[norm_filename]
            if replace_substring:
                path2_name = path2_name.replace(*replace_substring)
            path2 = os.path.join(folder2, path2_name)

            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                continue

            _, mask1 = cv2.threshold(img1, 1, 255, cv2.THRESH_BINARY)
            _, mask2 = cv2.threshold(img2, 1, 255, cv2.THRESH_BINARY)

            intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool)).sum()
            union = np.logical_or(mask1.astype(bool), mask2.astype(bool)).sum()
            iou = intersection / union if union else 0.0

            iou_scores.append(iou)
            filenames.append(filename)

    return np.array(iou_scores), filenames

def print_stats(label, iou_array):
    if len(iou_array) == 0:
        print(f"\nStats for {label}: No valid images compared.")
        return
    mean, median, std = np.mean(iou_array), np.median(iou_array), np.std(iou_array)
    over_50 = np.sum(iou_array > 0.5)
    over_95 = np.sum(iou_array > 0.95)
    total = len(iou_array)
    print(f"\nStats for {label}:")
    print(f"Mean IoU: {mean:.4f}")
    print(f"Median IoU: {median:.4f}")
    print(f"Std IoU: {std:.4f}")
    print(f">50% IoU: {over_50}/{total} ({over_50/total*100:.2f}%)")
    print(f">95% IoU: {over_95}/{total} ({over_95/total*100:.2f}%)")

def visualize_masks(human_a_path, comparison_path, comparison_name, replace_substring=None):
    path_human_a = os.path.join(human_a_path, comparison_path)
    if replace_substring:
        path_comparison = os.path.join(comparison_name, comparison_path.replace(*replace_substring))
    else:
        path_comparison = os.path.join(comparison_name, comparison_path)

    mask_a = cv2.imread(path_human_a, cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(path_comparison, cv2.IMREAD_GRAYSCALE)
    if mask_a is None or mask_b is None:
        print("Unable to load images for visualization.")
        return

    _, mask_a = cv2.threshold(mask_a, 1, 255, cv2.THRESH_BINARY)
    _, mask_b = cv2.threshold(mask_b, 1, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(8, 8))
    plt.title(f"Overlay of Masks (Red=Human A, Green={os.path.basename(comparison_name)})")
    plt.imshow(mask_a, cmap='Reds', alpha=0.5)
    plt.imshow(mask_b, cmap='Greens', alpha=0.5)
    plt.axis('off')
    plt.show()

# Input paths
folder_human_a = "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/2024-08-29-Morning/manual_sam2_niall"

# List of directories to compare with human_a
comparison_dirs = [
    # {
    #     "label": "Human B",
    #     "path": "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/2024-08-29-Morning/manual_sam2_niall",
    #     "replace_substring": None
    # },
    {
        "label": "Human A",
        "path": "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/2024-08-29-Morning/manual_sam2/manual_sam2_masks",
        "replace_substring": None
    },
    {
        "label": "Leaf Pipeline",
        "path": "/media/edward/HDD/Workspace/Resnet/LEAF/pipeline/all_images",
        "replace_substring": ("masked", "depth")
    },
    
    # {
    #     "label": "Whole LAM",
    #     "path": "/media/edward/HDD/Workspace/Resnet/WHOLE/lam/all_images",
    #     "replace_substring": ("masked", "depth")
    # },

    # Add more entries here if needed
]

# Loop through and compute stats
for comp in comparison_dirs:
    iou, filenames = compute_iou_scores_with_filenames(
        folder_human_a,
        comp["path"],
        image_conditions,
        replace_substring=comp["replace_substring"]
    )
    print_stats(comp["label"], iou)

    # Optional: Visualize median
    if len(iou) > 0:
        idx_median = np.abs(iou - np.median(iou)).argmin()
        representative_image_name = filenames[idx_median]
        print(f"Representative image for {comp['label']}: {representative_image_name}")
        visualize_masks(folder_human_a, representative_image_name, comp["path"], comp["replace_substring"])
