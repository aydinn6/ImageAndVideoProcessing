import tkinter as tk
import ImageProcessorGui

def open_kmeans_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("K-Means Parameters")  # Set popup window title
    popup.geometry("300x200")          # Set size of the popup window
    popup.resizable(False, False)      # Disable resizing the popup

    # Define default values for K-means parameters
    default_k = tk.StringVar(value="3")
    default_max_iter = tk.StringVar(value="100")
    default_epsilon = tk.StringVar(value="1.0")

    # Label and entry for number of clusters (k)
    tk.Label(popup, text="Number of clusters (k):").pack(pady=(10, 0))
    k_entry = tk.Entry(popup, textvariable=default_k)
    k_entry.pack()

    # Label and entry for max iterations
    tk.Label(popup, text="Max Iterations:").pack(pady=(10, 0))
    max_iter_entry = tk.Entry(popup, textvariable=default_max_iter)
    max_iter_entry.pack()

    # Label and entry for epsilon (convergence threshold)
    tk.Label(popup, text="Epsilon:").pack(pady=(10, 0))
    epsilon_entry = tk.Entry(popup, textvariable=default_epsilon)
    epsilon_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            k = int(k_entry.get())
            max_iter = int(max_iter_entry.get())
            epsilon = float(epsilon_entry.get())
            popup.destroy()  # Close the popup
            callback_func(k, max_iter, epsilon)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_k.get()), int(default_max_iter.get()), float(default_epsilon.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_binary_thresholding_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Binary Thresholding Parameters")   # Set popup window title
    popup.geometry("300x200")                       # Set size of the popup window
    popup.resizable(False, False)       # Disable resizing the popup

    # Define default values for Binary thresholding parameters
    default_binary_thresholding = tk.StringVar(value="127")

    # Label and entry for binary thresholding
    tk.Label(popup, text="Binary thresholding value:").pack(pady=(10, 0))
    binary_thresholding_entry = tk.Entry(popup, textvariable=default_binary_thresholding)
    binary_thresholding_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            binary_thresholding = int(binary_thresholding_entry.get())
            if ( (binary_thresholding > 0) and (binary_thresholding < 255) ):
                popup.destroy()  # Close the popup
                callback_func(binary_thresholding)  # Call the provided callback function with user inputs
            else:
                tk.messagebox.showerror("Error", "Please enter valid thresholding values. Between 0 and 255.")
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid thresholding values.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_binary_thresholding.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_clahe_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("CLAHE Parameters")  # Set popup window title
    popup.geometry("300x200")          # Set size of the popup window
    popup.resizable(False, False)      # Disable resizing the popup

    # Define default values for CLAHE parameters
    default_clipLimit = tk.StringVar(value="2.0")
    default_tileGridSizeX = tk.StringVar(value="8")
    default_tileGridSizeY = tk.StringVar(value="8")

    # Label and entry for Clip limit
    tk.Label(popup, text="Clip Limit:").pack(pady=(10, 0))
    clipLimit_entry = tk.Entry(popup, textvariable=default_clipLimit)
    clipLimit_entry.pack()

    # Label and entry for Tile grid size x
    tk.Label(popup, text="Tile Grid Size X:").pack(pady=(10, 0))
    tilGridSizeX_entry = tk.Entry(popup, textvariable=default_tileGridSizeX)
    tilGridSizeX_entry.pack()

    # Label and entry for Tile grid size Y
    tk.Label(popup, text="Tile Grid Size Y:").pack(pady=(10, 0))
    tilGridSizeY_entry = tk.Entry(popup, textvariable=default_tileGridSizeY)
    tilGridSizeY_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            clipLimit = float(clipLimit_entry.get())
            tilGridSizeX = int(tilGridSizeX_entry.get())
            tilGridSizeY = int(tilGridSizeY_entry.get())
            popup.destroy()  # Close the popup
            callback_func(clipLimit, tilGridSizeX, tilGridSizeY)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(float(default_clipLimit.get()), int(default_tileGridSizeX.get()), float(default_tileGridSizeY.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_log_transform_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Log Transform Parameters")  # Set popup window title
    popup.geometry("300x200")          # Set size of the popup window
    popup.resizable(False, False)      # Disable resizing the popup
    # Define default values for Log transformation parameters
    default_c_log = tk.StringVar(value="1.0")
    # Label and entry for Clip limit
    tk.Label(popup, text="Transformation constant:").pack(pady=(10, 0))
    c_log_entry = tk.Entry(popup, textvariable=default_c_log)
    c_log_entry.pack()
    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            c_log = float(c_log_entry.get())
            popup.destroy()  # Close the popup
            callback_func(c_log)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")
    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))
    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(float(default_c_log.get()))
        except:
            pass
        popup.destroy()
    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_power_law_transform_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Power-Law Transform Parameters")  # Set popup window title
    popup.geometry("300x200")          # Set size of the popup window
    popup.resizable(False, False)      # Disable resizing the popup
    # Define default values for Power-law parameters
    default_gama = tk.StringVar(value="0.4")
    default_c_power = tk.StringVar(value="1.0")
    # Label and entry for Gama Value
    tk.Label(popup, text="Gama Value:").pack(pady=(10, 0))
    gama_entry = tk.Entry(popup, textvariable=default_gama)
    gama_entry.pack()
    # Label and entry for Transformation Constant
    tk.Label(popup, text="Transformation constant:").pack(pady=(10, 0))
    c_power_entry = tk.Entry(popup, textvariable=default_c_power)
    c_power_entry.pack()
    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            gama    = float(gama_entry.get())
            c_power = float(c_power_entry.get())
            popup.destroy()  # Close the popup
            callback_func(gama, c_power)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")
    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))
    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(float(default_gama.get()),float(default_c_power.get()))
        except:
            pass
        popup.destroy()
    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_sobel_edge_detection_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Sobel Edge Detection Parameters")  # Set popup window title
    popup.geometry("300x200")          # Set size of the popup window
    popup.resizable(False, False)      # Disable resizing the popup
    # Define default values for Sobel Edge Detection parameters
    default_ksize = tk.StringVar(value="3")
    default_direction = tk.StringVar(value="both")
    # Label and entry for Kernel Size
    tk.Label(popup, text="Kernel Size Value:").pack(pady=(10, 0))
    ksize_entry = tk.Entry(popup, textvariable=default_ksize)
    ksize_entry.pack()
    # Label and entry for Direction
    tk.Label(popup, text="Direction(x,y and both):").pack(pady=(10, 0))
    direction_entry = tk.Entry(popup, textvariable=default_direction)
    direction_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            ksize = int(ksize_entry.get())
            direction = direction_entry.get()
            popup.destroy()  # Close the popup
            callback_func(ksize, direction)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")
    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))
    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_ksize.get()),default_direction.get())
        except:
            pass
        popup.destroy()
    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_canny_edge_detection_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Canny Edge Detection Parameters")  # Set popup window title
    popup.geometry("300x200")  # Set size of the popup window
    popup.resizable(False, False)  # Disable resizing the popup
    # Define default values for Canny Edge Detection parameters
    default_lower_threshold = tk.StringVar(value="100")
    default_upper_threshold = tk.StringVar(value="200")

    # Label and entry for Lower Threshold
    tk.Label(popup, text="Lower Threshold:").pack(pady=(10, 0))
    lower_threshold_entry = tk.Entry(popup, textvariable=default_lower_threshold)
    lower_threshold_entry.pack()

    # Label and entry for Upper Threshold
    tk.Label(popup, text="Upper Threshold:").pack(pady=(10, 0))
    upper_threshold_entry = tk.Entry(popup, textvariable=default_upper_threshold)
    upper_threshold_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            lower_threshold = int(lower_threshold_entry.get())
            upper_threshold = int(upper_threshold_entry.get())
            popup.destroy()  # Close the popup
            callback_func(lower_threshold, upper_threshold)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_lower_threshold.get()), int(default_upper_threshold.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_log_edge_detection_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Laplacian of Gaussian Edge Detection Parameters")  # Set popup window title
    popup.geometry("300x200")  # Set size of the popup window
    popup.resizable(False, False)  # Disable resizing the popup
    # Define default values for LoG Edge Detection parameters
    default_kernel_size = tk.StringVar(value="3")
    default_sigma = tk.StringVar(value="1.0")

    # Label and entry for Kernel Size
    tk.Label(popup, text="Kernel Size:").pack(pady=(10, 0))
    kernel_size_entry = tk.Entry(popup, textvariable=default_kernel_size)
    kernel_size_entry.pack()

    # Label and entry for Sigma
    tk.Label(popup, text="Sigma:").pack(pady=(10, 0))
    sigma_entry = tk.Entry(popup, textvariable=default_sigma)
    sigma_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            kernel_size = int(kernel_size_entry.get())
            sigma = float(sigma_entry.get())
            popup.destroy()  # Close the popup
            callback_func(kernel_size, sigma)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_kernel_size.get()), float(default_sigma.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_scharr_edge_detection_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Scharr Edge Detection Parameters")  # Set popup window title
    popup.geometry("300x200")  # Set size of the popup window
    popup.resizable(False, False)  # Disable resizing the popup
    # Define default values for Scharr Edge Detection parameters
    default_direction = tk.StringVar(value="both")
    default_ksize     = tk.StringVar(value="3")
    # Label and entry for Kernel Size
    tk.Label(popup, text="Kernel Size Value:").pack(pady=(10, 0))
    ksize_entry = tk.Entry(popup, textvariable=default_ksize)
    ksize_entry.pack()
    # Label and entry for Direction
    tk.Label(popup, text="Direction(x, y, and both):").pack(pady=(10, 0))
    direction_entry = tk.Entry(popup, textvariable=default_direction)
    direction_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs
            direction = direction_entry.get()
            ksize = int(ksize_entry.get())
            popup.destroy()  # Close the popup
            callback_func(ksize, direction)  # Call the provided callback function with default kernel size and user input
        except Exception:
            tk.messagebox.showerror("Error", "Please enter a valid direction value.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_ksize.get()), default_direction.get())  # Default kernel size 3
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_mean_filter_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Mean Filter Parameters")  # Set popup window title
    popup.geometry("300x200")  # Set size of the popup window
    popup.resizable(False, False)  # Disable resizing the popup
    # Define default value for kernel size
    default_kernel_size = tk.StringVar(value="3")

    # Label and entry for Kernel Size
    tk.Label(popup, text="Kernel Size (odd number):").pack(pady=(10, 0))
    kernel_size_entry = tk.Entry(popup, textvariable=default_kernel_size)
    kernel_size_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user input and convert to appropriate type
            kernel_size = int(kernel_size_entry.get())
            if kernel_size % 2 == 0:
                raise ValueError("Kernel size should be an odd number.")
            popup.destroy()  # Close the popup
            callback_func(kernel_size)  # Call the provided callback function with user input
        except Exception:
            tk.messagebox.showerror("Error", "Please enter a valid odd number for kernel size.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default value
    def on_close():
        try:
            callback_func(int(default_kernel_size.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_gauss_filter_popup(callback_func):
    # Create a new top-level popup window
    popup = tk.Toplevel()
    popup.title("Gauss Filter Parameters")  # Set popup window title
    popup.geometry("300x250")  # Set size of the popup window
    popup.resizable(False, False)  # Disable resizing the popup
    # Define default values for Gaussian Filter parameters
    default_kernel_size = tk.StringVar(value="3")
    default_sigma = tk.StringVar(value="1.0")

    # Label and entry for Kernel Size
    tk.Label(popup, text="Kernel Size (odd number):").pack(pady=(10, 0))
    kernel_size_entry = tk.Entry(popup, textvariable=default_kernel_size)
    kernel_size_entry.pack()

    # Label and entry for Sigma
    tk.Label(popup, text="Sigma (standard deviation):").pack(pady=(10, 0))
    sigma_entry = tk.Entry(popup, textvariable=default_sigma)
    sigma_entry.pack()

    # Callback function to be executed when the OK button is clicked
    def on_ok():
        try:
            # Get user inputs and convert to appropriate types
            kernel_size = int(kernel_size_entry.get())
            sigma = float(sigma_entry.get())
            if kernel_size % 2 == 0:
                raise ValueError("Kernel size should be an odd number.")
            popup.destroy()  # Close the popup
            callback_func(kernel_size, sigma)  # Call the provided callback function with user inputs
        except Exception:
            tk.messagebox.showerror("Error", "Please enter valid numeric values for kernel size and sigma.")

    # OK button
    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    # If user closes the window with the "X" button, use default values
    def on_close():
        try:
            callback_func(int(default_kernel_size.get()), float(default_sigma.get()))
        except:
            pass
        popup.destroy()

    # Bind the close event to the on_close function
    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_frequency_low_pass_filter_popup(callback_func):
    popup = tk.Toplevel()
    popup.title("Frequency Domain Low-Pass Filter Parameters")
    popup.geometry("300x180")
    popup.resizable(False, False)

    default_cutoff = tk.StringVar(value="30")

    tk.Label(popup, text="Cutoff Radius (e.g., 30):").pack(pady=(10, 0))
    cutoff_entry = tk.Entry(popup, textvariable=default_cutoff)
    cutoff_entry.pack()

    def on_ok():
        try:
            cutoff = int(cutoff_entry.get())
            popup.destroy()
            callback_func(cutoff)
        except:
            tk.messagebox.showerror("Error", "Please enter a valid integer value for cutoff radius.")

    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    def on_close():
        try:
            callback_func(int(default_cutoff.get()))
        except:
            pass
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_frequency_high_pass_filter_popup(callback_func):
    popup = tk.Toplevel()
    popup.title("Frequency Domain High-Pass Filter Parameters")
    popup.geometry("300x180")
    popup.resizable(False, False)

    default_cutoff = tk.StringVar(value="30")

    tk.Label(popup, text="Cutoff Radius (e.g., 30):").pack(pady=(10, 0))
    cutoff_entry = tk.Entry(popup, textvariable=default_cutoff)
    cutoff_entry.pack()

    def on_ok():
        try:
            cutoff = int(cutoff_entry.get())
            popup.destroy()
            callback_func(cutoff)
        except:
            tk.messagebox.showerror("Error", "Please enter a valid integer value for cutoff radius.")

    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    def on_close():
        try:
            callback_func(int(default_cutoff.get()))
        except:
            pass
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_frequency_bilinear_interpolation_popup(callback_func):
    popup = tk.Toplevel()
    popup.title("Bilinear Interpolation Parameters")
    popup.geometry("300x180")
    popup.resizable(False, False)

    default_width = tk.StringVar(value="640")
    default_height = tk.StringVar(value="480")

    tk.Label(popup, text="New Width (e.g., 640):").pack(pady=(10, 0))
    width_entry = tk.Entry(popup, textvariable=default_width)
    width_entry.pack()

    tk.Label(popup, text="New Height (e.g., 480):").pack(pady=(10, 0))
    height_entry = tk.Entry(popup, textvariable=default_height)
    height_entry.pack()

    def on_ok():
        try:
            new_width = int(width_entry.get())
            new_height = int(height_entry.get())
            popup.destroy()
            callback_func(new_width, new_height)
        except:
            tk.messagebox.showerror("Error", "Please enter valid integer values for width and height.")

    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    def on_close():
        try:
            callback_func(int(default_width.get()), int(default_height.get()))
        except:
            pass
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_frequency_nearest_neighbour_interpolation_popup(callback_func):
    popup = tk.Toplevel()
    popup.title("Nearest Neighbor Interpolation Parameters")
    popup.geometry("300x180")
    popup.resizable(False, False)

    default_width = tk.StringVar(value="640")
    default_height = tk.StringVar(value="480")

    tk.Label(popup, text="New Width (e.g., 640):").pack(pady=(10, 0))
    width_entry = tk.Entry(popup, textvariable=default_width)
    width_entry.pack()

    tk.Label(popup, text="New Height (e.g., 480):").pack(pady=(10, 0))
    height_entry = tk.Entry(popup, textvariable=default_height)
    height_entry.pack()

    def on_ok():
        try:
            new_width = int(width_entry.get())
            new_height = int(height_entry.get())
            popup.destroy()
            callback_func(new_width, new_height)
        except:
            tk.messagebox.showerror("Error", "Please enter valid integer values for width and height.")

    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    def on_close():
        try:
            callback_func(int(default_width.get()), int(default_height.get()))
        except:
            pass
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_close)

def open_frequency_bicubic_interpolation_popup(callback_func):
    popup = tk.Toplevel()
    popup.title("Bicubic Interpolation Parameters")
    popup.geometry("300x180")
    popup.resizable(False, False)

    default_width = tk.StringVar(value="640")
    default_height = tk.StringVar(value="480")

    tk.Label(popup, text="New Width (e.g., 640):").pack(pady=(10, 0))
    width_entry = tk.Entry(popup, textvariable=default_width)
    width_entry.pack()

    tk.Label(popup, text="New Height (e.g., 480):").pack(pady=(10, 0))
    height_entry = tk.Entry(popup, textvariable=default_height)
    height_entry.pack()

    def on_ok():
        try:
            new_width = int(width_entry.get())
            new_height = int(height_entry.get())
            popup.destroy()
            callback_func(new_width, new_height)
        except:
            tk.messagebox.showerror("Error", "Please enter valid integer values for width and height.")

    tk.Button(popup, text="OK", command=on_ok).pack(pady=(15, 5))

    def on_close():
        try:
            callback_func(int(default_width.get()), int(default_height.get()))
        except:
            pass
        popup.destroy()

    popup.protocol("WM_DELETE_WINDOW", on_close)

def on_transform_select(*args):
    transformation = selected_transform.get()
    ImageProcessorGui.apply_transformation(transformation)

# Create the main window
root = tk.Tk()
# Set the window title
root.title("Image Processor GUI")
# Set the window size
root.geometry("500x450")
# Set background color for the root window
root.config(bg="lightblue")
# Set window size editability
root.resizable(False, False)

# Create a textbox (Entry widget) for displaying the file path
file_path_entry = tk.Entry(root, width=40)
file_path_entry.place(x=100, y=14)

# Create a 'Browse' button
browse_button = tk.Button(root, text="Browse", command=ImageProcessorGui.browse_image, height=1, width=8)
browse_button.place(x=355, y=13)
# Create a 'View Image' button
viewImage_button = tk.Button(root, text="View Image", command=ImageProcessorGui.view_image, height=1, width=10)
viewImage_button.place(x=170, y=410)
#Create a 'save Image' button
saveImage_button = tk.Button(root, text="Save Image", command=ImageProcessorGui.save_image, height=1, width=10)
saveImage_button.place(x=260, y=410)
#Create a 'Otsu Thresholding' button
otsu_button = tk.Button(root, text="Apply Otsu", command=ImageProcessorGui.apply_otsu, height=1, width=10)
otsu_button.place(x=23, y=93)
#Create a 'K-Means Clustering Thresholding' button
kmeans_cluster_button = tk.Button(root, text="Apply K-Means Clustering", command=ImageProcessorGui.kmeans_input_getter_popup, height=1, width=20)
kmeans_cluster_button.place(x=23, y=121)
#Create a 'Binary Thresholding' button
binary_thresholding_button = tk.Button(root, text='Apply Binary Thresholding', command=ImageProcessorGui.binary_thresholding_input_getter_popup, height=1, width=20)
binary_thresholding_button.place(x=23, y=149)
#Create a 'CLAHE' button
clahe_button = tk.Button(root, text='Apply CLAHE', command=ImageProcessorGui.clahe_input_getter_popup, height=1, width=20)
clahe_button.place(x=23, y=235)
#Create a 'GHE' button
ghe_button = tk.Button(root, text='Apply GHE', command=ImageProcessorGui.apply_ghe, height=1, width=20)
ghe_button.place(x=23, y=263)
#Create a 'Log Transform' button
log_transform_button = tk.Button(root, text='Apply Log Transform', command=ImageProcessorGui.log_transform_input_getter_popup, height=1, width=20)
log_transform_button.place(x=23, y=291)
#Create a 'Power-Law Transform' button
power_law_transform_button = tk.Button(root, text='Apply Power-Law Transform', command=ImageProcessorGui.power_law_transform_input_getter_popup, height=1, width=20)
power_law_transform_button.place(x=23, y=319)
#Create a 'Sobel Edge Detection' button
sobel_edge_detection_button = tk.Button(root, text='Apply Sobel Edge Detection', command=ImageProcessorGui.sobel_edge_detection_input_getter_popup, height=1, width=20)
sobel_edge_detection_button.place(x=180, y=93)
#Create a 'Canny Edge Detection' button
canny_edge_detection_button = tk.Button(root, text='Apply Canny Edge Detection', command=ImageProcessorGui.canny_edge_detection_input_getter_popup, height=1, width=20)
canny_edge_detection_button.place(x=180, y=121)
#Create a 'Laplacian of Gaussian Edge Detection' button
log_edge_detection_button = tk.Button(root, text='Apply LoG Edge Detection', command=ImageProcessorGui.log_edge_detection_input_getter_popup, height=1, width=20)
log_edge_detection_button.place(x=180, y=149)
#Create a 'Scharr Edge Detection' button
scharr_edge_detection_button = tk.Button(root, text='Apply Scharr Edge Detection', command=ImageProcessorGui.scharr_edge_detection_input_getter_popup, height=1, width=20)
scharr_edge_detection_button.place(x=180, y=177)
#Create a 'Mean Filter' button
mean_filter_button = tk.Button(root, text='Apply Mean Filter', command=ImageProcessorGui.mean_filter_input_getter_popup, height=1, width=20)
mean_filter_button.place(x=180, y=235)
#Create a 'Gauss Filter' button
gauss_filter_button = tk.Button(root, text='Apply Gauss Filter', command=ImageProcessorGui.gauss_filter_input_getter_popup, height=1, width=20)
gauss_filter_button.place(x=180, y=263)
#Create a 'Low Pass Filter' button
low_pass_filter_button = tk.Button(root, text='Apply Low Pass Filter', command=ImageProcessorGui.frequency_low_pass_filter_input_getter_popup, height=1, width=20)
low_pass_filter_button.place(x=180, y=319)
#Create a 'High Pass Filter' button
low_pass_filter_button = tk.Button(root, text='Apply High Pass Filter', command=ImageProcessorGui.frequency_high_pass_filter_input_getter_popup, height=1, width=20)
low_pass_filter_button.place(x=180, y=346)
#Create a 'Bilinear Interpolation' button
bilinear_interpolation_button = tk.Button(root, text='Apply Bilinear Interpolation', command=ImageProcessorGui.bilinear_interpolation_input_getter_popup, height=1, width=20)
bilinear_interpolation_button.place(x=337, y=93)
#Create a 'Nearest Neighbour Interpolation' button
nneighbour_interpolation_button = tk.Button(root, text='NNeighbour Interpolation', command=ImageProcessorGui.nearest_neighbour_interpolation_input_getter_popup, height=1, width=20)
nneighbour_interpolation_button.place(x=337, y=121)
#Create a 'Bicubic Neighbour Interpolation' button
bicubic_interpolation_button = tk.Button(root, text='Apply Bicubic Interpolation', command=ImageProcessorGui.bicubic_interpolation_input_getter_popup, height=1, width=20)
bicubic_interpolation_button.place(x=337, y=149)
# List of transformations
transform_list = [
    "RGB to Gray", "Gray to RGB",
    "RGB to HSI", "HSI to RGB",
    "RGB to YUV", "YUV to RGB",
    "Fourier Transform", "Inverse Fourier Transform",
    "Discrete Cosine Transform", "Inverse Discrete Cosine Transform"
]
# Dropdown menu for transformation selection
selected_transform = tk.StringVar()
selected_transform.set(transform_list[0])  # Default value
selected_transform.trace("w", on_transform_select)
transform_menu = tk.OptionMenu(root, selected_transform, *transform_list)
transform_menu.place(x=350, y=235)


#Image processing methodologies label
methodologies_label = tk.Label(root, text="Methodologies:", fg="red", bg="lightblue" ,font=("Arial", 10, "bold"), height=2, width=20)
methodologies_label.place(x=160, y=37)
#Segmentation label
segmentation_label = tk.Label(root, text="Segmentations:", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
segmentation_label.place(x=0, y=65)
#Histogram Equalization and Image Enhancement
histogram_equalization_label = tk.Label(root, text="Histogram Equalization and\nImage Enhancement Techniques:", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=24)
histogram_equalization_label.place(x=15, y=205)
#Edge Detection Label
edge_detection_label = tk.Label(root, text="Edge Detection", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
edge_detection_label.place(x=175, y=65)
#Time Domain Filtering Operations Label
time_dmn_filtering_ops_label = tk.Label(root, text="Time Domain\nFiltering Operations", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
time_dmn_filtering_ops_label.place(x=175, y=205)
#Frequency Domain Filtering Operations Label
freq_dmn_filtering_ops_label = tk.Label(root, text="Frequency Domain\nFiltering Operations", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
freq_dmn_filtering_ops_label.place(x=175, y=291)
#Select Input Image label
input_label = tk.Label(root, text="Select\n Input Image:", fg="red", bg="lightblue" ,font=("Arial", 8, "bold"), height=2, width=10)
input_label.place(x=10, y=10)
#Image Rescaling Operations
img_rescaling_ops_label = tk.Label(root, text="Image Rescaling Operations", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
img_rescaling_ops_label.place(x=350, y=65)
#Image Domain Transformation Operations
img_dmn_transformation_label = tk.Label(root, text="Image Domain\nTransformations", fg="red", bg="lightblue" ,font=("Arial", 7, "bold"), height=2, width=20)
img_dmn_transformation_label.place(x=350, y=205)
