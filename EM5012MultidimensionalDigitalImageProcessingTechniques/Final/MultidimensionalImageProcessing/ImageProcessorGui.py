from tkinter import filedialog, messagebox
import CreateGui
import tkinter as tk
import cv2
import numpy as np
from math import pi

GlobalImage = None
Fourier_shifted_img = None
Window_counter = 0

def browse_image():
    # Open file dialog to choose an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    if file_path:
        try:
            # Try to read the image with OpenCV
            img = cv2.imread(file_path)
            # If the image has 3 channels (BGR)
            if img.ndim == 3 and img.shape[2] == 3:
                # Check if the image is grayscale
                if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2]):
                    # If the BGR channels are the same, it's a grayscale image
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            global GlobalImage
            GlobalImage = img
            if img is None:
                raise ValueError("Invalid image file")
            # If image is valid, insert the file path into the textbox
            CreateGui.file_path_entry.delete(0, tk.END)  # Clear the textbox
            CreateGui.file_path_entry.insert(0, file_path)  # Insert the file path
        except Exception as e:
            # If there's an error (like invalid file), show a popup
            messagebox.showerror("Error", "Please select a valid image")

def view_image():
    try:
        # Check if GlobalImage is None, meaning no image is loaded
        if GlobalImage is None:
            raise ValueError("Invalid image file") # Raise an error if no image is available
        else:
            # Generate a unique window name for each image view
            global Window_counter
            cv2.imshow(f"ImageWindow_{Window_counter}",GlobalImage)
            Window_counter += 1 # Increment the counter to ensure each window has a unique name
    except Exception as e:
        # Show an error message if something goes wrong (e.g., no image to display)
        messagebox.showerror("Error", "There is no image to display")

def save_image():
    try:
        # Check if GlobalImage is None, meaning no image is loaded
        if GlobalImage is None:
            raise ValueError("Invalid image file") # Raise an error if no image is available
        else:
            # Open file saver window
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"),
                                                                ("PNG files", "*.png"),
                                                                ("All files", "*.*")],
                                                     title="Save Image")
            if file_path:
                cv2.imwrite(file_path, GlobalImage)
            else:
                raise ValueError("Invalid image file")
    except Exception as e:
        # Show an error message if something goes wrong (e.g., no image to save)
        messagebox.showerror("Error", "There is no image to save")

def apply_otsu():
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            _, GlobalImage = cv2.threshold(GlobalImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            messagebox.showinfo("Otsu Threshold", "Otsu Thresholding is applied succesfully")
    except Exception as e:
        messagebox.showerror("Error", "Otsu thresholding can not be applied.\nPlease check the input image. \nIt should be grayscale.")

def kmeans_input_getter_popup():
    CreateGui.open_kmeans_popup(apply_kmeans_clustering)

def apply_kmeans_clustering(k, max_iter, epsilon):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Reshape the image to a 2D array (each pixel will be a row)
            pixels = GlobalImage.reshape((-1, 3))

            _, labels, centers = cv2.kmeans(np.float32(pixels), k, None,
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon),
                                             10, cv2.KMEANS_RANDOM_CENTERS)
            # Convert the cluster centers to 8-bit format
            centers = np.uint8(centers)
            # Recolor each pixel according to the cluster it belongs to
            kmeans_clustered_img = centers[labels.flatten()]
            # Reshape the image back to its original 3D format (for displaying purposes)
            GlobalImage = kmeans_clustered_img.reshape(GlobalImage.shape)
            messagebox.showinfo("K-Means Clustering Threshold", "K-Means Clustering Thresholding is applied succesfully")
    except Exception as e:
        messagebox.showerror("Error",
                          "K-Means Clustering thresholding can not be applied.\nPlease check the input image. \nIt should be RGB.")

def binary_thresholding_input_getter_popup():
    CreateGui.open_binary_thresholding_popup(apply_binary_threshold)

def apply_binary_threshold(threshold_value):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply binary thresholding
            _, GlobalImage = cv2.threshold(GlobalImage, threshold_value, 255, cv2.THRESH_BINARY)
            messagebox.showinfo("Binary Threshold", "Binary Thresholding is applied successfully")
    except Exception as e:
        messagebox.showerror("Error", "Binary thresholding cannot be applied.\nPlease check the input image.\nIt should be grayscale.")

def rgb2hsi(img):
    bgr = np.int32(cv2.split(img))/255
    blue = bgr[0]
    green = bgr[1]
    red = bgr[2]
    intensity = np.divide(blue + green + red, 3)
    minim = np.minimum(np.minimum(red, green), blue)
    saturation = 1 - 3 * np.divide(minim, red + green + blue+1.e-17)
    sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))
    hue = np.arccos((0.5 * ((red - green) + (red - blue)) / (sqrt_calc + 1.e-17)))
    hue[(blue>green)] = 2*pi - hue[(blue>green)]
    hue = hue / (2*pi)
    hue[(saturation==0)] = 0
    hsi = cv2.merge((hue, saturation, intensity))
    return hsi, hue, saturation, intensity

def hsi2rgb(H, S, I):
    H = H * 2 * np.pi
    R, G, B = np.zeros(H.shape), np.zeros(H.shape), np.zeros(H.shape) # values will be between [0,1]

    idx=np.where((0<=H)&(H<2*np.pi/3+1.e-15))
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    idx = np.where((2 * pi / 3 <= H) & (H < 4 * np.pi / 3))
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]-2*pi/3) / np.cos(pi - H[idx]))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    idx = np.where((4 * pi / 3 <= H) & (H <= 2 * np.pi))
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx] - 4 * pi / 3) / np.cos(5 * pi / 3 - H[idx]))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    R = np.maximum(np.minimum(R, 1), 0) * 255  #clipping into [0 1] range then to [0 255]
    G = np.maximum(np.minimum(G, 1), 0) * 255
    B = np.maximum(np.minimum(B, 1), 0) * 255
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)
    RGB = cv2.merge((B, G, R))
    return RGB

def clahe_input_getter_popup():
    CreateGui.open_clahe_popup(apply_clahe)

def apply_clahe(clipLimit, tileGridSizeX, tileGridSizeY):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Contrast Limited Adaptive Histogram Equalization
            hsi, hue, saturation, intensity = rgb2hsi(GlobalImage)
            # Natural logarithm is applied to the Intensity component
            # Describe c constant for natural logarithm
            intensity = intensity * 255
            c = c = 1 / np.log(1 + np.max(intensity))
            intensity_ln = c * np.log(1 + intensity)  # Natural logarithm of I component
            intensity_ln255 = cv2.normalize(intensity_ln, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_32F)
            intensity_ln255 = intensity_ln255.astype(np.uint8)
            # Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSizeX, tileGridSizeY))
            CLAHEintensity = clahe.apply(intensity_ln255)
            CLAHEintensity = np.divide(CLAHEintensity, 255, dtype=float)  # to the range [0 1]
            GlobalImage = hsi2rgb(hue, saturation, CLAHEintensity)
            messagebox.showinfo("Contrast Limited Adaptive Histogram Equalization", "CLAHE is applied successfully")
    except Exception as e:
        messagebox.showerror("Error", "CLAHE cannot be applied.\nPlease check the input image.\nIt should be RGB.")

def apply_ghe():
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Histogram Equalization
            GlobalImage = cv2.equalizeHist(GlobalImage)
            messagebox.showinfo("General Histogram Equalization", "General Histogram equalization is applied successfully")
    except Exception as e:
        messagebox.showerror("Error", "GHE cannot be applied.\nPlease check the input image.\nIt should be Grayscale.")

def log_transform_input_getter_popup():
    CreateGui.open_log_transform_popup(apply_log_transform)

def apply_log_transform(c_log):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            GlobalImage = np.float32(GlobalImage)
            GlobalImage = c_log * np.log(1 + GlobalImage)  # Log transform of image
            GlobalImage = cv2.normalize(GlobalImage, None, 0, 255, cv2.NORM_MINMAX)
            GlobalImage = np.uint8(GlobalImage)
            messagebox.showinfo("Log Transformation","Log transformation is applied successfully")
    except Exception as e:
        messagebox.showerror("Error", "Log transformation cannot be applied.\nPlease check the input image.\nIt should be Grayscale.")

def power_law_transform_input_getter_popup():
    CreateGui.open_power_law_transform_popup(apply_power_law_transform)

def apply_power_law_transform(gama, c_power_law):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            GlobalImage = c_power_law * np.power(GlobalImage, gama)  # Power law transformation of image
            GlobalImage = cv2.normalize(GlobalImage, None, 0, 255, cv2.NORM_MINMAX)
            GlobalImage = np.uint8(GlobalImage)
            messagebox.showinfo("Power-Law Transformation","Log transformation is applied successfully")
    except Exception as e:
        messagebox.showerror("Error","Power-Law transformation cannot be applied.\nPlease check the input image.\nIt should be Grayscale.")

def sobel_edge_detection_input_getter_popup():
    CreateGui.open_sobel_edge_detection_popup(apply_sobel_edge_detection)

def apply_sobel_edge_detection(kernel_size, direction):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Sobel operator in the specified direction
            if direction == 'x':
                sobel_x = cv2.Sobel(GlobalImage, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobel_y = np.zeros_like(sobel_x)  # No vertical edge detection
            elif direction == 'y':
                sobel_x = np.zeros_like(GlobalImage)  # No horizontal edge detection
                sobel_y = cv2.Sobel(GlobalImage, cv2.CV_64F, 0, 1, ksize=kernel_size)
            else:  # 'both'
                sobel_x = cv2.Sobel(GlobalImage, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobel_y = cv2.Sobel(GlobalImage, cv2.CV_64F, 0, 1, ksize=kernel_size)
            # Convert gradients to absolute values
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)
            # Combine X and Y gradients
            sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
            # Update global image
            GlobalImage = sobel_combined
            messagebox.showinfo("Sobel Edge Detection", "Sobel edge detection has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error",f"Sobel edge detection could not be applied.\nPlease ensure the input image is in grayscale.")

def canny_edge_detection_input_getter_popup():
    CreateGui.open_canny_edge_detection_popup(apply_canny_edge_detection)

def apply_canny_edge_detection(lower_threshold, upper_threshold):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Canny Edge Detection
            edges = cv2.Canny(GlobalImage, lower_threshold, upper_threshold)
            # Update the global image with the edges
            GlobalImage = edges
            messagebox.showinfo("Canny Edge Detection", "Canny edge detection has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Canny edge detection could not be applied.\nPlease ensure the input image is in grayscale.")

def log_edge_detection_input_getter_popup():
    CreateGui.open_log_edge_detection_popup(apply_log_edge_detection)

def apply_log_edge_detection(kernel_size, sigma):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(GlobalImage, (kernel_size, kernel_size), sigma)
            # Apply Laplacian operator on the blurred image
            log_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
            # Convert to absolute value
            log_image = cv2.convertScaleAbs(log_image)
            # Update the global image
            GlobalImage = log_image
            messagebox.showinfo("LoG Edge Detection", "Laplacian of Gaussian edge detection has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"LoG edge detection could not be applied.\nPlease ensure the input image is in grayscale.")

def scharr_edge_detection_input_getter_popup():
    CreateGui.open_scharr_edge_detection_popup(apply_scharr_edge_detection)

def apply_scharr_edge_detection(kernel_size, direction):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Scharr operator in the specified direction
            if direction == 'x':
                scharr_x = cv2.Scharr(GlobalImage, cv2.CV_64F, 1, 0)
                scharr_y = np.zeros_like(scharr_x)  # No vertical edge detection
            elif direction == 'y':
                scharr_x = np.zeros_like(GlobalImage)  # No horizontal edge detection
                scharr_y = cv2.Scharr(GlobalImage, cv2.CV_64F, 0, 1)
            else:  # 'both'
                scharr_x = cv2.Scharr(GlobalImage, cv2.CV_64F, 1, 0)
                scharr_y = cv2.Scharr(GlobalImage, cv2.CV_64F, 0, 1)

            # Convert gradients to absolute values
            abs_scharr_x = cv2.convertScaleAbs(scharr_x)
            abs_scharr_y = cv2.convertScaleAbs(scharr_y)
            # Combine X and Y gradients
            scharr_combined = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)
            # Update the global image
            GlobalImage = scharr_combined
            messagebox.showinfo("Scharr Edge Detection", "Scharr edge detection has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error",
                             f"Scharr edge detection could not be applied.\nPlease ensure the input image is in grayscale.")

def mean_filter_input_getter_popup():
    CreateGui.open_mean_filter_popup(apply_mean_filter)

def apply_mean_filter(kernel_size):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Mean (Average) Filter using cv2's blur function
            mean_filtered_image = cv2.blur(GlobalImage, (kernel_size, kernel_size))
            # Update the global image
            GlobalImage = mean_filtered_image
            messagebox.showinfo("Mean Filter", "Mean filter has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Mean filter could not be applied.\nPlease ensure the input image is in grayscale.")

def gauss_filter_input_getter_popup():
    CreateGui.open_gauss_filter_popup(apply_gauss_filter)

def apply_gauss_filter(kernel_size, sigma):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")
        else:
            # Apply Gaussian Filter using cv2's GaussianBlur function
            gauss_filtered_image = cv2.GaussianBlur(GlobalImage, (kernel_size, kernel_size), sigma)
            # Update the global image
            GlobalImage = gauss_filtered_image
            messagebox.showinfo("Gauss Filter", "Gauss filter has been applied successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Gauss filter could not be applied.\nPlease ensure the input image is in grayscale.")

def frequency_low_pass_filter_input_getter_popup():
    CreateGui.open_frequency_low_pass_filter_popup(apply_frequency_low_pass_filter)

def apply_frequency_low_pass_filter(cutoff):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        # Görüntü boyutlarını al
        rows, cols = GlobalImage.shape
        crow, ccol = rows // 2, cols // 2

        # 2D low-pass filtre maskesi oluştur
        mask = np.zeros((rows, cols), dtype=np.float32)
        cv2.circle(mask, (ccol, crow), cutoff, 1, -1)

        # FFT, filtre, ters FFT işlemleri
        f = np.fft.fft2(GlobalImage)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        GlobalImage = np.uint8(GlobalImage)
        # Filtrelenmiş görüntüyi geri at
        GlobalImage = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

        messagebox.showinfo("Low-Pass Filter", "Low-pass filter applied successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"Low-pass filter could not be applied.\nPlease ensure the input image is in grayscale.")

def frequency_high_pass_filter_input_getter_popup():
    CreateGui.open_frequency_high_pass_filter_popup(apply_frequency_high_pass_filter)

def apply_frequency_high_pass_filter(cutoff):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        # Get image dimensions
        rows, cols = GlobalImage.shape
        crow, ccol = rows // 2, cols // 2

        # Create a 2D high-pass filter mask
        mask = np.ones((rows, cols), dtype=np.float32)
        cv2.circle(mask, (ccol, crow), cutoff, 0, -1)

        # FFT, filter, inverse FFT operations
        f = np.fft.fft2(GlobalImage)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        GlobalImage = np.uint8(GlobalImage)
        # Update the GlobalImage with the filtered image
        GlobalImage = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

        messagebox.showinfo("High-Pass Filter", "High-pass filter applied successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"High-pass filter could not be applied.\nPlease ensure the input image is in grayscale.")

def bilinear_interpolation_input_getter_popup():
    CreateGui.open_frequency_bilinear_interpolation_popup(apply_bilinear_interpolation)

def apply_bilinear_interpolation(new_width, new_height):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        # Resize the image using bilinear interpolation
        resized_image = cv2.resize(GlobalImage, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Update the GlobalImage with the resized image
        GlobalImage = resized_image

        messagebox.showinfo("Bilinear Interpolation", "Image resized successfully using bilinear interpolation.")

    except Exception as e:
        messagebox.showerror("Error", f"Bilinear interpolation could not be applied.\nError: {str(e)}")

def nearest_neighbour_interpolation_input_getter_popup():
    CreateGui.open_frequency_nearest_neighbour_interpolation_popup(apply_nearest_neighbour_interpolation)

def apply_nearest_neighbour_interpolation(new_width, new_height):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        # Resize the image using nearest neighbor interpolation
        resized_image = cv2.resize(GlobalImage, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Update the GlobalImage with the resized image
        GlobalImage = resized_image

        messagebox.showinfo("Nearest Neighbor Interpolation", "Image resized successfully using nearest neighbor interpolation.")

    except Exception as e:
        messagebox.showerror("Error", f"Nearest neighbor interpolation could not be applied.\nError: {str(e)}")

def bicubic_interpolation_input_getter_popup():
    CreateGui.open_frequency_bicubic_interpolation_popup(apply_bicubic_interpolation)

def apply_bicubic_interpolation(new_width, new_height):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        # Resize the image using bicubic interpolation
        resized_image = cv2.resize(GlobalImage, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Update the GlobalImage with the resized image
        GlobalImage = resized_image

        messagebox.showinfo("Bicubic Interpolation", "Image resized successfully using bicubic interpolation.")

    except Exception as e:
        messagebox.showerror("Error", f"Bicubic interpolation could not be applied.\nError: {str(e)}")

def apply_transformation(selected_transform):
    global GlobalImage
    try:
        if GlobalImage is None:
            raise ValueError("Invalid image file")

        if selected_transform == "RGB to Gray":
            apply_rgb_to_gray()
        elif selected_transform == "Gray to RGB":
            apply_gray_to_rgb()
        elif selected_transform == "RGB to HSI":
            apply_rgb_to_hsi()
        elif selected_transform == "HSI to RGB":
            apply_hsi_to_rgb()
        elif selected_transform == "RGB to YUV":
            apply_rgb_to_yuv()
        elif selected_transform == "YUV to RGB":
            apply_yuv_to_rgb()
        elif selected_transform == "Fourier Transform":
            apply_fourier_transform()
        elif selected_transform == "Inverse Fourier Transform":
            apply_inverse_fourier_transform()
        elif selected_transform == "Discrete Cosine Transform":
            apply_dct()
        elif selected_transform == "Inverse Discrete Cosine Transform":
            apply_inverse_dct()

        messagebox.showinfo("Transformation Applied", f"{selected_transform} applied successfully.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while applying the transformation: {str(e)}")

def apply_rgb_to_gray():
    global GlobalImage
    GlobalImage = cv2.cvtColor(GlobalImage, cv2.COLOR_BGR2GRAY)

def apply_gray_to_rgb():
    global GlobalImage
    rgb_image = np.zeros((*GlobalImage.shape, 3), dtype=np.uint8)
    for i in range(GlobalImage.shape[0]):
        for j in range(GlobalImage.shape[1]):
            y = GlobalImage[i, j]
            # Scale each channel differently to add color tone variation
            r = min(255, int(y * 1.2))  # Boost red
            g = min(255, int(y * 0.9))  # Moderate green
            b = min(255, int(y * 0.7))  # Lower blue
            rgb_image[i, j] = [b, g, r]  # OpenCV uses BGR format
    GlobalImage = cv2.cvtColor(GlobalImage, cv2.COLOR_GRAY2BGR)
    GlobalImage = rgb_image

def apply_rgb_to_hsi():
    global GlobalImage
    GlobalImage, hue, saturation, intensity = rgb2hsi(GlobalImage)

def apply_hsi_to_rgb():
    global GlobalImage
    hue = GlobalImage[:,:,0]
    saturation = GlobalImage[:,:,1]
    intensity = GlobalImage[:,:,2]
    GlobalImage = hsi2rgb(hue, saturation, intensity)

def apply_rgb_to_yuv():
    global GlobalImage
    GlobalImage = cv2.cvtColor(GlobalImage, cv2.COLOR_BGR2YUV)

def apply_yuv_to_rgb():
    global GlobalImage
    GlobalImage = cv2.cvtColor(GlobalImage, cv2.COLOR_YUV2BGR)

def apply_fourier_transform():
    global GlobalImage
    global Fourier_shifted_img
    GlobalImage = cv2.dft(np.float32(GlobalImage), flags=cv2.DFT_COMPLEX_OUTPUT)
    GlobalImage = np.fft.fftshift(GlobalImage)
    Fourier_shifted_img = GlobalImage
    GlobalImage = 20*np.log(cv2.magnitude(GlobalImage[:,:,0],GlobalImage[:,:,1]))
    GlobalImage = cv2.normalize(GlobalImage, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

def apply_inverse_fourier_transform():
    global GlobalImage
    global Fourier_shifted_img
    if Fourier_shifted_img is None:
        raise ValueError("Fourier transformation should be applied before")
    else:
        GlobalImage = np.fft.ifftshift(Fourier_shifted_img)
        GlobalImage = cv2.idft(GlobalImage)
        GlobalImage = cv2.magnitude(GlobalImage[:, :, 0], GlobalImage[:, :, 1])
        GlobalImage = cv2.normalize(GlobalImage, None, 0, 255, cv2.NORM_MINMAX)
        GlobalImage = np.uint8(GlobalImage)

def apply_dct():
    global GlobalImage
    GlobalImage = cv2.dct(np.float32(GlobalImage))

def apply_inverse_dct():
    global GlobalImage
    GlobalImage = cv2.idct(np.float32(GlobalImage))
    GlobalImage = np.uint8(GlobalImage)

if __name__ == "__main__":
    CreateGui.root.mainloop()


