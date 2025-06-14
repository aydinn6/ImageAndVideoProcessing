# Setup
import numpy as np
from skimage.feature import corner_peaks
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from utils import get_output_space, warp_image
from panorama import harris_corners, match_descriptors, describe_keypoints, ransac, hog_descriptor

# Load 8 grayscale images
images = [rgb2gray(imread(f'frame{i}.jpg')) for i in range(1, 8)]

# Step 1: Detect keypoints and extract descriptors
keypoints_list = []
descriptors_list = []

for img in images:
    keypoints = corner_peaks(harris_corners(img, window_size=3),
                                                 threshold_rel=0.05,
                                                 exclude_border=8)
    descriptors = describe_keypoints(img, keypoints,
                                     desc_func=hog_descriptor,
                                     patch_size=16)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

# Step 2: Compute homographies with RANSAC
homographies = [np.eye(3)]  # Identity for the first image

for i in range(len(images) - 1):
    matches = match_descriptors(descriptors_list[i], descriptors_list[i+1], 0.7)
    H, _ = ransac(keypoints_list[i], keypoints_list[i+1], matches, threshold=1)
    # Chain homographies to the previous
    homographies.append(homographies[-1] @ H)

# Step 3: Compute final output shape and offset
output_shape, offset = get_output_space(images[0], images[1:], homographies[1:])

# Step 4: Warp all images into the panorama space
warped_images = []
masks = []

for i in range(len(images)):
    H = homographies[i]
    warped = warp_image(images[i], H, output_shape, offset)
    mask = (warped != -1)
    warped[~mask] = 0
    warped_images.append(warped)
    masks.append(mask)

# Step 5: Merge all warped images
merged = np.zeros(output_shape)
overlap = np.zeros(output_shape)

for warped, mask in zip(warped_images, masks):
    merged += warped
    overlap += mask.astype(float)

# Step 6: Normalize final panorama
normalized = merged / np.maximum(overlap, 1)

# Show result
plt.figure(figsize=(15, 8))
plt.imshow(normalized, cmap='gray')
plt.axis('off')
plt.title("HOG + RANSAC Generated Panorama with 8 Images")
plt.show()
