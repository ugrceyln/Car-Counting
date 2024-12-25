import cv2
import numpy as np
from math import exp, pi

def create_roi_mask(frame, x_start, x_end, y_start, y_end):
    """Create a mask for the region of interest (road area)"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = 255
    return mask

def create_gaussian_kernel(kernel_size, sigma=0):
    """
    Create a Gaussian kernel with the specified size and sigma.
    If sigma is 0, it's computed using OpenCV's formula.
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # If sigma is 0, compute it based on kernel size (OpenCV's formula)
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Calculate center point
    center = kernel_size // 2

    # Create kernel
    kernel = np.zeros((kernel_size, kernel_size))

    # Calculate normalization constant
    norm = 1 / (2 * pi * sigma ** 2)

    # Fill kernel with Gaussian values
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = norm * exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize kernel so sum equals 1
    return kernel / kernel.sum()

def gaussian_blur(image, kernel_size, sigma=0):
    """
    Optimized Gaussian blur implementation using NumPy's vectorized operations.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Pad the image
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Prepare for convolution
    height, width = image.shape

    # Use NumPy's as_strided to create sliding windows
    from numpy.lib.stride_tricks import as_strided

    # Calculate strides for sliding window
    window_shape = (height, width, kernel_size, kernel_size)
    window_strides = (padded_image.strides[0], padded_image.strides[1],
                      padded_image.strides[0], padded_image.strides[1])

    # Create sliding windows
    windows = as_strided(padded_image, shape=window_shape, strides=window_strides)

    # Apply convolution using vectorized operations
    output = np.sum(windows * kernel, axis=(2, 3))

    return output.astype(np.uint8)

def bitwise_and(src1, src2, mask=None):
    """
    Perform bitwise AND operation between two images with optional mask.
    """
    # Convert inputs to numpy arrays if they aren't already
    src1 = np.asarray(src1)
    src2 = np.asarray(src2)

    # Perform bitwise AND between src1 and src2
    result = np.bitwise_and(src1, src2)

    # If mask is provided, apply it
    if mask is not None:
        # Convert mask to boolean array (0s become False, non-zero becomes True)
        mask = np.asarray(mask, dtype=bool)

        # Verify mask dimensions match the input images
        if mask.shape != src1.shape:
            raise ValueError("Mask dimensions must match input image dimensions")

        # Apply mask (set pixels to 0 where mask is False)
        result = np.where(mask, result, 0)

    return result.astype(np.uint8)

def absdiff(src1, src2):
    """
    Calculate the per-element absolute difference between two arrays.
    """
    # Convert inputs to numpy arrays if they aren't already
    src1 = np.asarray(src1)
    src2 = np.asarray(src2)

    # Check if shapes match
    if src1.shape != src2.shape:
        raise ValueError("Input arrays must have the same shape")

    # Calculate absolute difference
    # This is equivalent to: |src1 - src2|
    diff = np.abs(src1.astype(np.int16) - src2.astype(np.int16))

    # Convert back to uint8 (8-bit unsigned integer)
    return diff.astype(np.uint8)

def threshold(src, thresh_value, max_value):
    """
    Apply thresholding to an image.
    """
    # Convert input to numpy array if it isn't already
    src = np.asarray(src)

    # Create output array
    dst = np.zeros_like(src, dtype=np.uint8)

    # Apply binary threshold
    dst[src > thresh_value] = max_value
    dst[src <= thresh_value] = 0


    return thresh_value, dst.astype(np.uint8)

def create_rectangular_kernel(size):
    """
    Create a rectangular kernel of ones with given size.
    """
    if isinstance(size, int):
        size = (size, size)
    return np.ones(size, dtype=np.uint8)

def erode(src, kernel, iterations=1):
    """
    Optimized erosion implementation using NumPy's vectorized operations.
    """
    if len(src.shape) != 2:
        raise ValueError("Input image must be 2D (grayscale/binary)")

    if kernel is None:
        kernel = create_rectangular_kernel(3)

    result = src.copy()

    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Prepare for sliding windows
    from numpy.lib.stride_tricks import as_strided

    for _ in range(iterations):
        # Pad the image
        padded = np.pad(result, ((pad_height, pad_height), (pad_width, pad_width)),
                        mode='constant', constant_values=255)

        # Calculate window shape and strides
        window_shape = (result.shape[0], result.shape[1], kernel_height, kernel_width)
        window_strides = (padded.strides[0], padded.strides[1],
                          padded.strides[0], padded.strides[1])

        # Create sliding windows
        windows = as_strided(padded, shape=window_shape, strides=window_strides)

        # Apply kernel and get minimum values
        if np.all(kernel == 1):
            # Optimized path for rectangular kernel
            result = np.min(windows, axis=(2, 3))
        else:
            # Handle custom kernels
            kernel_mask = kernel.reshape(1, 1, kernel_height, kernel_width)
            masked_windows = np.where(kernel_mask, windows, 255)
            result = np.min(masked_windows, axis=(2, 3))

    return result.astype(np.uint8)

def dilate(src, kernel, iterations=1):
    """
    Optimized dilation implementation using NumPy's vectorized operations.
    """
    if len(src.shape) != 2:
        raise ValueError("Input image must be 2D (grayscale/binary)")

    if kernel is None:
        kernel = create_rectangular_kernel(3)

    result = src.copy()

    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2

    # Prepare for sliding windows
    from numpy.lib.stride_tricks import as_strided

    for _ in range(iterations):
        # Pad the image
        padded = np.pad(result, ((pad_height, pad_height), (pad_width, pad_width)),
                        mode='constant', constant_values=0)

        # Calculate window shape and strides
        window_shape = (result.shape[0], result.shape[1], kernel_height, kernel_width)
        window_strides = (padded.strides[0], padded.strides[1],
                          padded.strides[0], padded.strides[1])

        # Create sliding windows
        windows = as_strided(padded, shape=window_shape, strides=window_strides)

        # Apply kernel and get maximum values
        if np.all(kernel == 1):
            # Optimized path for rectangular kernel
            result = np.max(windows, axis=(2, 3))
        else:
            # Handle custom kernels
            kernel_mask = kernel.reshape(1, 1, kernel_height, kernel_width)
            masked_windows = np.where(kernel_mask, windows, 0)
            result = np.max(masked_windows, axis=(2, 3))

    return result.astype(np.uint8)

def find_contours_from_scratch(binary_image):
    """
    Finds contours in a binary image from scratch with improved stability.
    :param binary_image: Binary image (numpy array of 0s and 255s).
    :return: List of contours in OpenCV format.
    """
    contours = []
    visited = np.zeros_like(binary_image, dtype=bool)  # To track visited pixels

    # More robust neighbor offsets for finding contour points
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def is_border_pixel(x, y):
        """Determine if a pixel is on the border of an object."""
        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < binary_image.shape[1] and 0 <= ny < binary_image.shape[0]:
                if binary_image[ny, nx] == 0:
                    return True
        return False

    def trace_contour(start_point):
        """Trace a contour starting from a given point."""
        contour = []
        x, y = start_point
        current_point = (x, y)

        # Use a maximum path length to prevent infinite loops
        max_path_length = binary_image.size // 10

        while len(contour) < max_path_length:
            contour.append(current_point)
            visited[current_point[1], current_point[0]] = True

            found_next = False
            for dx, dy in neighbor_offsets:
                nx, ny = current_point[0] + dx, current_point[1] + dy

                # Check boundaries and pixel conditions
                if (0 <= nx < binary_image.shape[1] and
                        0 <= ny < binary_image.shape[0] and
                        binary_image[ny, nx] == 255 and
                        not visited[ny, nx]):

                    # Additional stability check: prefer border pixels
                    if is_border_pixel(nx, ny):
                        current_point = (nx, ny)
                        found_next = True
                        break

            # If no border pixel found, find any unvisited white pixel
            if not found_next:
                for dx, dy in neighbor_offsets:
                    nx, ny = current_point[0] + dx, current_point[1] + dy

                    if (0 <= nx < binary_image.shape[1] and
                            0 <= ny < binary_image.shape[0] and
                            binary_image[ny, nx] == 255 and
                            not visited[ny, nx]):
                        current_point = (nx, ny)
                        found_next = True
                        break

            # If no unvisited pixels, contour is complete
            if not found_next:
                break

        return contour

    # Improved contour finding algorithm
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            # Only start tracing from border pixels
            if (binary_image[y, x] == 255 and
                    not visited[y, x] and
                    is_border_pixel(x, y)):

                contour = trace_contour((x, y))

                # Filter out very small or noisy contours
                if len(contour) > 5:
                    # Convert to OpenCV format: [[[x, y]], [[x, y]], ...]
                    opencv_contour = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
                    contours.append(opencv_contour)

    return contours

def otsu_thresholding(grayscale_image):
    """
    Perform Otsu's thresholding to find optimal image segmentation
    """

    # Ensure image is flattened for histogram calculation
    pixels = grayscale_image.flatten()

    # Calculate histogram
    histogram = np.zeros(256, dtype=int)
    for pixel in pixels:
        histogram[int(pixel)] += 1

    # Total number of pixels
    total_pixels = len(pixels)

    # Calculate cumulative histogram and normalized histogram
    cumulative_histogram = np.cumsum(histogram)
    probability = histogram / total_pixels

    # Variables to store optimal threshold
    max_variance = 0
    optimal_threshold = 0

    # Iterate through all possible threshold values
    for threshold in range(1, 256):
        # Separate background and foreground
        background_pixels = cumulative_histogram[threshold - 1]
        foreground_pixels = total_pixels - background_pixels

        # Skip if no pixels in either group
        if background_pixels == 0 or foreground_pixels == 0:
            continue

        # Probabilities of background and foreground
        background_prob = background_pixels / total_pixels
        foreground_prob = 1 - background_prob

        # Mean calculation
        background_mean = np.sum(np.arange(threshold) * histogram[:threshold]) / background_pixels
        foreground_mean = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / foreground_pixels

        # Between-class variance calculation (Otsu's method)
        variance = background_prob * foreground_prob * (background_mean - foreground_mean) ** 2

        # Update optimal threshold
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold

    # Create thresholded image
    thresholded_image = np.zeros_like(grayscale_image)

    if optimal_threshold < 25 or optimal_threshold > 45:
        optimal_threshold = 30
    thresholded_image[grayscale_image >= optimal_threshold] = 255

    return thresholded_image, optimal_threshold

def bounding_rect_nested(contour):
    """
    Calculate bounding rectangle for a nested coordinate array.

    Parameters:
    contour (list): Nested list of coordinates

    Returns:
    tuple: (x_min, y_min, width, height)
    """
    # Convert nested list to clean 2D numpy array of coordinates
    coords = np.array([point[0] for point in contour])

    # Calculate minimum and maximum coordinates
    x_min = np.min(coords[:, 0])
    y_min = np.min(coords[:, 1])
    x_max = np.max(coords[:, 0])
    y_max = np.max(coords[:, 1])

    # Calculate width and height
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return (x_min, y_min, width, height)

def contour_area_nested(contour):
    """
    Calculate the area of a contour using the Shoelace formula (Green's area formula).
    """
    # Convert nested list to clean 2D numpy array of coordinates
    coords = np.array([point[0] for point in contour])

    # Shoelace formula implementation
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]

    # Take the absolute value and divide by 2
    area = abs(area) / 2.0

    return area
