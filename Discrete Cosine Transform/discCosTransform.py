import numpy as np
import cv2
import os
from scipy.fft import dct, idct

# Read the image
img_path = '...'  # Path to the original image
img = cv2.imread(img_path, 0)

row, col = img.shape[:2]  # Get the dimensions of the image
block_size = 8  # Block size for DCT

# Quantization matrix for less aggressive compression
q_50 = np.array([[4, 3, 2, 4, 6, 10, 12, 14],
                 [3, 3, 4, 5, 7, 15, 16, 14],
                 [4, 4, 5, 7, 10, 14, 17, 14],
                 [4, 5, 6, 8, 14, 22, 20, 16],
                 [5, 6, 9, 13, 16, 26, 24, 18],
                 [6, 9, 13, 16, 20, 26, 29, 23],
                 [12, 15, 18, 20, 24, 30, 30, 25],
                 [18, 22, 24, 26, 28, 25, 26, 24]])

# Pad the image to make its dimensions a multiple of 8
new_row = (row // 8 + 1) * 8
new_col = (col // 8 + 1) * 8
padded_img = np.zeros((new_row, new_col), dtype=np.uint8)
padded_img[:row, :col] = img

# Empty array to hold the quantized DCT coefficients
result_img = np.zeros((new_row, new_col), dtype=np.float32)

# Apply DCT and quantization block by block
for i in range(0, new_row, block_size):
    for j in range(0, new_col, block_size):
        block = padded_img[i:i + block_size, j:j + block_size]

        # Perform DCT on the block
        dct_img = dct(block, type=2, norm='ortho', axis=0)
        dct2 = dct(dct_img, type=2, norm='ortho', axis=1)

        # Quantize the DCT coefficients
        quantized_block = np.round(dct2 / q_50)  # Quantized block
        result_img[i:i + block_size, j:j + block_size] = quantized_block

# Inverse DCT (dequantization and reconstruction)
reconstructed_img = np.zeros((new_row, new_col), dtype=np.float32)

for i in range(0, new_row, block_size):
    for j in range(0, new_col, block_size):
        block = result_img[i:i + block_size, j:j + block_size]

        # Perform inverse DCT to reconstruct the block
        idct_img = idct(block, type=2, norm='ortho', axis=0)
        idct2 = idct(idct_img, type=2, norm='ortho', axis=1)

        reconstructed_img[i:i + block_size, j:j + block_size] = idct2

# Normalize the reconstructed image to 0-255 range
reconstructed_img = np.clip(reconstructed_img, 0, 255)  # Clipping to the valid pixel range
reconstructed_img = reconstructed_img.astype(np.uint8)  # Convert to uint8 format

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Compressed and Reconstructed Image', reconstructed_img)

# Save the final reconstructed image in the same directory as the original image
# Construct the path for the final image
final_image_path = os.path.join(os.path.dirname(img_path), 'compressed_reconstructed_image.jpg')

# Save the final image
cv2.imwrite(final_image_path, reconstructed_img)

# Print the path where the final image is saved
print(f"Final image saved at: {final_image_path}")

# Wait for a key press and close the display windows
cv2.waitKey(0)
cv2.destroyAllWindows()
