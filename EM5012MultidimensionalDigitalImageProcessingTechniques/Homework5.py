import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
# ========================
# Question 1: (a)
# ========================
def apply_filter(image, kernel):
    """
    Function to apply a spatial filter.
    :param image: The image to be filtered (grayscale or color).
    :param kernel: The filter matrix (kernel) to be applied.
    :return: The filtered image.
    """
    return cv2.filter2D(image, -1, kernel)
# Select images (grayscale and color images)
root = tk.Tk()
root.withdraw()
# Open the file selector window
file_path_rgb = filedialog.askopenfilename(
    title="Bir renkli görüntü dosyası seçin",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)
# Write selected file path
if file_path_rgb:
    print(f"Seçilen dosya: {file_path_rgb}")
    try:
        # Open the image and check color domain
        color_image = Image.open(file_path_rgb)
        if color_image.mode != 'RGB':
            messagebox.showerror("Hata", "Seçilen dosya RGB formatında değil. Lütfen RGB formatında bir dosya seçin.")
            exit()  # Exit the program
    except Exception as e:
        messagebox.showerror("Hata", f"Görüntü açılamadı: {e}")
        exit()  # Exit the program
else:
    print("Hiçbir dosya seçilmedi.")
    exit() # Exit the program

# Open the file selector window
file_path_gray = filedialog.askopenfilename(
    title="Bir gri görüntü dosyası seçin",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)
# Write selected file path
if file_path_gray:
    print(f"Seçilen dosya: {file_path_gray}")
    try:
        # Open the image and check color domain
        gray_image = Image.open(file_path_gray)
        if gray_image.mode != 'L':  # Check Grayscale (L) mode
            messagebox.showerror("Hata",
                                 "Seçilen dosya Grayscale formatında değil. Lütfen Grayscale formatında bir dosya seçin.")
            exit()  # Exit the program
    except Exception as e:
        messagebox.showerror("Hata", f"Görüntü açılamadı: {e}")
        exit()  # Exit the program
else:
    print("Hiçbir dosya seçilmedi.")
    exit() # Exit the program
root.destroy()
color_image = cv2.cvtColor(np.array(color_image), cv2.COLOR_RGB2BGR)
gray_image = np.array(gray_image)

# Average filters (3x3, 9x9, 15x15)
kernels = [np.ones((3, 3), np.float32) / 9,
           np.ones((9, 9), np.float32) / 81,
           np.ones((15, 15), np.float32) / 225]
# Apply filters and visualize the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, kernel in enumerate(kernels):
    # For grayscale image
    filtered_gray = apply_filter(gray_image, kernel)
    axs[0, i].imshow(filtered_gray, cmap='gray')
    axs[0, i].set_title(f'Gray {kernel.shape[0]}x{kernel.shape[1]} Filter')
    # For color image
    filtered_color = apply_filter(color_image, kernel)
    axs[1, i].imshow(cv2.cvtColor(filtered_color, cv2.COLOR_BGR2RGB))
    axs[1, i].set_title(f'Color {kernel.shape[0]}x{kernel.shape[1]} Filter')
plt.tight_layout()
plt.show()
# ========================
# Question 1: (b)
# ========================
def salt_and_pepper_noise(image, density):
    """
    Function that adds salt and pepper noise to an image.
    :param image: The input image.
    :param density: The noise density (between 0.0 and 1.0).
    :return: The noisy image.
    """
    noisy_image = image.copy()
    total_pixels = noisy_image.size
    num_salt = int(density * total_pixels * 0.5)
    num_pepper = int(density * total_pixels * 0.5)
    # Add salt noise
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    # Add pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_image
# Add salt and pepper noise
noisy_gray = salt_and_pepper_noise(gray_image, density=0.15)
noisy_color = salt_and_pepper_noise(color_image, density=0.15)
# Display the noisy images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(noisy_gray, cmap='gray')
axs[0].set_title('Noisy Grayscale Image')
axs[1].imshow(cv2.cvtColor(noisy_color, cv2.COLOR_BGR2RGB))
axs[1].set_title('Noisy Color Image')
plt.tight_layout()
plt.show()
# ========================
# Question 1: (c)
# ========================
# Applying filter
filtered_noisy_gray = apply_filter(noisy_gray, kernels[1])
filtered_noisy_color = apply_filter(noisy_color, kernels[1])
# Display the filtered images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(filtered_noisy_gray, cmap='gray')
axs[0].set_title('Noisy Grayscale Image - 9x9 Filter')
axs[1].imshow(cv2.cvtColor(filtered_noisy_color, cv2.COLOR_BGR2RGB))
axs[1].set_title('Noisy Color Image - 9x9 Filter')
plt.tight_layout()
plt.show()
# ========================
# Question 1: (d)
# ========================
def median_filter(image, kernel_size):
    """
    Function that applies a median filter to the image.
    :param image: The input image.
    :param kernel_size: The neighborhood size (e.g., 3x3, 5x5).
    :return: The image filtered with the median filter.
    """
    return cv2.medianBlur(image, kernel_size)
# Apply median filter (3x3 neighborhood)
median_filtered_gray = median_filter(noisy_gray, 3)
# Display the resulting image
plt.figure(figsize=(6, 6))
plt.imshow(median_filtered_gray, cmap='gray')
plt.title('Median Filter - 3x3 Neighborhood')
plt.show()
# ========================
# Question 2:
# ========================
# Apply the Sobel operator to calculate the gradients in the X and Y directions
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in the X direction (horizontal edges)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in the Y direction (vertical edges)
# Compute the gradient magnitude (edge strength)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
# Visualize the Sobel gradients and original image
plt.figure(figsize=(12, 6))
# Original grayscale image
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
# Sobel X direction (horizontal edges)
plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X (Horizontal Edges)')
plt.axis('off')
# Sobel Y direction (vertical edges)
plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y (Vertical Edges)')
plt.axis('off')
plt.tight_layout()
plt.show()
# Visualize the Sobel gradient magnitude (edge strength)
plt.figure(figsize=(6, 6))
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Gradient Magnitude')
plt.axis('off')
plt.show()
# ========================
# Question 3:
# ========================
def convert_to_grayscale(image):
    """
    Converts a color image to grayscale using two methods:
    1. Average of RGB channels
    2. Weighted sum based on standard conversion formula
    """
    # Split the image into R, G, B channels
    (B, G, R) = cv2.split(image)
    # Method 1: Average of R, G, B channels
    gray_avg = np.mean(image, axis=2).astype(np.uint8)
    # Method 2: Standard weighted sum (0.2989*R + 0.5870*G + 0.1140*B)
    gray_weighted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_avg, gray_weighted
# Convert the color image to grayscale using the two methods
gray_avg, gray_weighted = convert_to_grayscale(color_image)
# Visualize the results
plt.figure(figsize=(12, 6))
# Display the first grayscale image (average of RGB channels)
plt.subplot(1, 2, 1)
plt.imshow(gray_avg, cmap='gray')
plt.title('Grayscale (Average of RGB)')
plt.axis('off')
# Display the second grayscale image (weighted sum conversion)
plt.subplot(1, 2, 2)
plt.imshow(gray_weighted, cmap='gray')
plt.title('Grayscale (Weighted Sum: 0.2989R + 0.5870G + 0.1140B)')
plt.axis('off')
plt.tight_layout()
plt.show()
# ========================
# Question 4:
# ========================
def histogram_equalization_color(img):
    # Convert from BGR to RGB (OpenCV reads images in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Split the image into individual color channels (Red, Green, Blue)
    r, g, b = cv2.split(img_rgb)
    # Plot histograms before histogram equalization for each channel
    plt.figure(figsize=(12, 6))
    # Plot histograms for Red, Green, and Blue channels before equalization
    plt.subplot(2, 3, 1)
    plt.hist(r.ravel(), bins=256, color='red', alpha=0.7)
    plt.title('Red Histogram (Before)')
    plt.subplot(2, 3, 2)
    plt.hist(g.ravel(), bins=256, color='green', alpha=0.7)
    plt.title('Green Histogram (Before)')
    plt.subplot(2, 3, 3)
    plt.hist(b.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Blue Histogram (Before)')
    # Apply histogram equalization to each color channel
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    # Merge the equalized channels back into a single image
    img_eq = cv2.merge([r_eq, g_eq, b_eq])
    # Plot histograms after histogram equalization for each channel
    plt.subplot(2, 3, 4)
    plt.hist(r_eq.ravel(), bins=256, color='red', alpha=0.7)
    plt.title('Red Histogram (After)')
    plt.subplot(2, 3, 5)
    plt.hist(g_eq.ravel(), bins=256, color='green', alpha=0.7)
    plt.title('Green Histogram (After)')
    plt.subplot(2, 3, 6)
    plt.hist(b_eq.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Blue Histogram (After)')
    # Adjust layout to avoid overlap of plots
    plt.tight_layout()
    plt.show()
    # Visualize the equalized color image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_eq)
    plt.title('Equalized Color Image')
    plt.axis('off')
    plt.show()
# Run the function with an image
histogram_equalization_color(color_image)