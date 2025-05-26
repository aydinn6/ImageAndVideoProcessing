import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image

# ========================
# Question 1:
# ========================
# Hide the main window
root = tk.Tk()
root.withdraw()
# Open the file selector window
file_path = filedialog.askopenfilename(
    title="Bir görüntü dosyası seçin",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)
# Write selected file path
if file_path:
    print(f"Seçilen dosya: {file_path}")
    img = Image.open(file_path)
else:
    print("Hiçbir dosya seçilmedi.")

if img.mode == 'RGB':
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
elif img.mode == 'L':
    img = np.array(img)

cv2.imshow("Original Image", img)
# ========================
# Question 1: (a)
# ========================
flipped_image_vertical = cv2.flip(img, 0)
cv2.imshow("Vertically Flipped Image", flipped_image_vertical)
# ========================
# Question 1: (b)
# ========================
mirrored_image = cv2.flip(img, 1)
cv2.imshow("Mirrored Image", mirrored_image)
# ========================
# Question 1: (c)
# ========================
negative_image = 255 - img
cv2.imshow("Negative Image", negative_image)
# ========================
# Question 1: (d)
# ========================
choice = simpledialog.askstring("Select Downscale Option",
                                "1. Enter the dowscale factor (e.g., 0.5). \n2. Enter the new dimensions (e.g., 800x600). \n\nPlease enter your choice (1 or 2):")
if choice == '1':
    # Get the scaling factor
    scale_factor = simpledialog.askfloat("Downscale Factor", "Enter the scaling factor (a value between 0 and 1):")
    if scale_factor is not None:
        # Calculate the new dimensions
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        new_size = (new_width, new_height)
        # Resize the image to the new dimensions
        downscaled_image = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
        messagebox.showinfo("Success", f"The image has been downscaled by a factor of {scale_factor} using Nearest Neighbor interpolation!")
elif choice == '2':
    # Get the new dimensions (e.g., 800x600 format)
    size_input = simpledialog.askstring("New Dimensions", "Enter the new dimensions (widthxheight):")
    if size_input:
        # Split the input and convert to integers
        try:
            new_width, new_height = map(int, size_input.split('x'))
            new_size = (new_width, new_height)
            # Resize the image to the new dimensions
            downscaled_image = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
            messagebox.showinfo("Success", f"The image has been downscaled to {new_width}x{new_height} using Nearest Neighbor interpolation!")
        except ValueError:
            messagebox.showerror("Error", "Invalid dimensions format! For example: 800x600")
else:
    messagebox.showerror("Invalid Option", "You entered an invalid option.")
cv2.imshow("Downsampled Image", downscaled_image)
# ========================
# Question 1: (e)
# ========================
upscale_choice = simpledialog.askstring("Select Upscale Option",
                                        "1. Enter the upscale factor (e.g., 2.0).\n2. Enter the new dimensions (e.g., 1600x1200). \n\nPlease enter your choice (1 or 2):")
if upscale_choice == '1':
      # Get the upscale factor
      upscale_factor = simpledialog.askfloat("Upscale Factor", "Enter the upscale factor (greater than 1):")
      if upscale_factor is not None and upscale_factor > 1:
          # Calculate the new dimensions based on upscale factor
          new_width = int(downscaled_image.shape[1] * upscale_factor)
          new_height = int(downscaled_image.shape[0] * upscale_factor)
          new_size = (new_width, new_height)
          # Resize the image using Nearest Neighbor interpolation
          upscaled_image = cv2.resize(downscaled_image, new_size, interpolation=cv2.INTER_NEAREST)
          messagebox.showinfo("Success", f"The image has been upscaled by a factor of {upscale_factor} using Nearest Neighbor interpolation!")
          cv2.imshow("Upscaled Image", upscaled_image)
elif upscale_choice == '2':
    # Get the new dimensions for upscale (e.g., 1600x1200 format)
    size_input = simpledialog.askstring("New Dimensions", "Enter the new dimensions (widthxheight):")
    if size_input:
        # Split the input and convert to integers
        try:
            new_width, new_height = map(int, size_input.split('x'))
            new_size = (new_width, new_height)
            # Resize the image using Nearest Neighbor interpolation
            upscaled_image = cv2.resize(downscaled_image, new_size, interpolation=cv2.INTER_NEAREST)
            messagebox.showinfo("Success", f"The image has been upscaled to {new_width}x{new_height} using Nearest Neighbor interpolation!")
            cv2.imshow("Upscaled Image", upscaled_image)
        except ValueError:
            messagebox.showerror("Error", "Invalid dimensions format! For example: 1600x1200")
else:
    messagebox.showerror("Invalid Option", "You entered an invalid option.")
cv2.imshow("Upscaled Image", upscaled_image)
# ========================
# Question 2:
# ========================
brightened_img = cv2.add(img, 50)
cv2.imshow("Brightened Image (Brightness +50)", brightened_img)
darkened_img = cv2.subtract(img, 75)
cv2.imshow("Darkened Image (Brightness -75)", darkened_img)
# ========================
# Question 3:
# ========================
c_log = 1
log_transformed_img = np.float32(img)
log_transformed_img = c_log * np.log(1 + log_transformed_img)       #Log transform of image
log_transformed_normalized = cv2.normalize(log_transformed_img, None, 0, 255, cv2.NORM_MINMAX)
log_transformed_normalized = np.uint8(log_transformed_normalized)
cv2.imshow("Log Transformed Image", log_transformed_normalized)

gama = 1.8
c_power_law = 0.8
power_law_transformed_img = np.float32(img)
power_law_transformed_img = c_power_law * np.power(img,gama)        #Power law transformation of image
power_law_transformed_normalized = cv2.normalize(power_law_transformed_img, None, 0, 255, cv2.NORM_MINMAX)
power_law_transformed_normalized = np.uint8(power_law_transformed_normalized)
cv2.imshow("Power-Law Tranformed Image", power_law_transformed_normalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
