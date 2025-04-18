import cv2
import numpy as np


def harris_corner_detector(image, k=0.04, threshold=0.01, window_size=3):
    """
    Implement Harris Corner Detection from scratch

    Args:
        image: Input image
        k: Harris detector free parameter
        threshold: Threshold for corner detection
        window_size: Size of the window for corner detection

    Returns:
        corners: List of (x, y) corner coordinates
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float for calculations
    gray = np.float32(gray)

    # Get image dimensions
    height, width = gray.shape

    # Calculate image gradients using Sobel
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of derivatives for the Harris matrix
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Create empty corner response map
    corner_response = np.zeros_like(gray)

    # Define offset for window
    offset = window_size // 2

    # Calculate Harris response for each pixel
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Extract window for current pixel
            window_Ixx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            window_Iyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            window_Ixy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]

            # Sum up elements in the window
            sum_Ixx = np.sum(window_Ixx)
            sum_Iyy = np.sum(window_Iyy)
            sum_Ixy = np.sum(window_Ixy)

            # Calculate determinant and trace
            det = (sum_Ixx * sum_Iyy) - (sum_Ixy**2)
            trace = sum_Ixx + sum_Iyy

            # Calculate corner response R = det(M) - k*(trace(M))^2
            R = det - k * (trace**2)

            corner_response[y, x] = R

    # Normalize corner response
    cv2.normalize(corner_response, corner_response, 0, 1, cv2.NORM_MINMAX)

    # Apply non-maximum suppression
    corners = []
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # If pixel is above threshold and is local maximum
            if corner_response[y, x] > threshold:
                # Check if it's a local maximum in 3x3 neighborhood
                window = corner_response[y-1:y+2, x-1:x+2]
                if corner_response[y, x] == np.max(window):
                    corners.append((x, y))

    return corners


def extract_msop_descriptors(image, corners, patch_size=41):
    """
    Extract Multi-Scale Oriented Patches (MSOP) descriptors

    Args:
        image: Input image
        corners: List of corner coordinates (x, y)
        patch_size: Size of the patch to extract (should be odd)

    Returns:
        descriptors: List of MSOP descriptors for each corner
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create Gaussian kernel for blurring
    sigma = 0.5 * (patch_size // 2)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Get image dimensions
    height, width = gray.shape

    # Half patch size for boundary checks
    half_patch = patch_size // 2

    # Store descriptors
    descriptors = []
    valid_corners = []

    for x, y in corners:
        # Skip corners too close to the image boundary
        if x < half_patch or x >= width - half_patch or y < half_patch or y >= height - half_patch:
            continue

        # Extract patch
        patch = blurred[y - half_patch:y + half_patch +
                        1, x - half_patch:x + half_patch + 1]

        # Normalize patch (zero mean and unit variance)
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-7)

        # Flatten patch to create descriptor
        descriptor = patch.flatten()

        # Add to list
        descriptors.append(descriptor)
        valid_corners.append((x, y))

    return descriptors, valid_corners


def detect_features(image):
    """
    Detect features in an image using Harris corner detector and MSOP descriptors

    Args:
        image: Input image

    Returns:
        features: Dictionary containing keypoints and descriptors
    """
    # Detect corners using Harris
    corners = harris_corner_detector(image)

    # Extract MSOP descriptors
    descriptors, valid_corners = extract_msop_descriptors(image, corners)

    # Return features
    return {
        'keypoints': valid_corners,
        'descriptors': descriptors
    }
