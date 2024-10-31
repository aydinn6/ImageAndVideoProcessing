import numpy as np
from PIL import Image

#A function is written for the interpolation kernel formula
def kernel_func(s, a):
    abs_s = abs(s)
    if 0 <= abs_s <= 1:
        result = (a + 2) * (abs_s ** 3) - (a + 3) * (abs_s ** 2) + 1
    elif 1 < abs_s <= 2:
        result = a * (abs_s ** 3) - (5 * a) * (abs_s ** 2) + (8 * a) * abs_s - 4 * a
    else:
        result = 0

    return result

def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :] = img

    # Fill the first and last two columns and rows
    zimg[2:H + 2, 0:2, :] = img[:, 0:1, :]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :] = img[0:1, :, :]

    # Complete the missing 8 points
    zimg[0:2, 0:2, :] = img[0, 0, :]
    zimg[H + 2:H + 4, 0:2, :] = img[H - 1, 0, :]
    zimg[H + 2:H + 4, W + 2:W + 4, :] = img[H - 1, W - 1, :]
    zimg[0:2, W + 2:W + 4, :] = img[0, W - 1, :]

    return zimg

def bicubic(img, ratio, a):
    # Read image size. For colored images, C channel is 3.
    H, W, C = img.shape

    # Send to padding function. Image is enlarged.
    img = padding(img, H, W, C)

    # Create an empty new image with appropriate size for output.
    dH = int(H * ratio)
    dW = int(W * ratio)

    # Create an empty matrix using numpy's np.zeros function.
    dst = np.zeros((dH, dW, 3))
    h = 1 / ratio

    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                # Get the coordinates of nearby values
                x = i * h + 2
                y = j * h + 2

                x1 = 1 + x - np.floor(x)
                x2 = x - np.floor(x)
                x3 = np.floor(x) + 1 - x
                x4 = np.floor(x) + 2 - x

                y1 = 1 + y - np.floor(y)
                y2 = y - np.floor(y)
                y3 = np.floor(y) + 1 - y
                y4 = np.floor(y) + 2 - y

                # Get the kernel matrix for the x direction
                mat_l = np.array([kernel_func(x1, a), kernel_func(x2, a), kernel_func(x3, a), kernel_func(x4, a)])

                # Get the kernel matrix for the y direction
                mat_r = np.array([[kernel_func(y1, a)], [kernel_func(y2, a)], [kernel_func(y3, a)], [kernel_func(y4, a)]])

                # Get the nearby 16 values
                mat_m = np.array([
                    [img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c],
                     img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
                    [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c],
                     img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
                    [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c],
                     img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
                    [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c],
                     img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]
                ])

                # Calculate the dot product
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

    return dst

# Main execution
img = Image.open('image.jpg')
img = np.array(img)  # Convert to numpy array

# Scaling factor
ratio = 2
# a coefficient
a = -1/2

# Apply bicubic interpolation
dst = bicubic(img, ratio, a)
print('Interpolation applied.')

# Print image sizes
print('Initial image size:', img.shape)
print('Final image size:', dst.shape)

# Save the output image
dst = np.clip(dst, 0, 255).astype(np.uint8)  # Clip and convert to uint8
output_img = Image.fromarray(dst)
output_img.save('bicubic_sonuc.png')

