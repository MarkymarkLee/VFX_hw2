import os
import cv2
import numpy as np


def harris_corner_detector(image, k=0.05, threshold=0.01, window_size=5):
    """
    Implement Harris Corner Detection from scratch

    Args:
        image: Input image
        k: Harris detector free parameter
        threshold: Threshold for corner detection
        window_size: Size of the window for corner detection. Should be odd.

    Returns:
        corners: List of (x, y) corner coordinates
    """

    assert window_size % 2 == 1, "Window size must be odd"

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Convert to float for calculations, normalize to [0, 1]
    gray = np.float32(gray) / 255.0

    # Get image dimensions
    height, width = gray.shape

    # Calculate image gradients using Sobel (implemented from scratch)
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Apply Sobel filters to get gradients
    Ix = cv2.filter2D(gray, -1, sobel_x)
    Iy = cv2.filter2D(gray, -1, sobel_y)

    # Compute products of derivatives for the Harris matrix
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    window = np.ones((window_size, window_size),
                     dtype=np.float32) / (window_size * window_size)

    # Apply Gaussian filter to the products of derivatives
    Ixx = cv2.filter2D(Ixx, -1, window)
    Iyy = cv2.filter2D(Iyy, -1, window)
    Ixy = cv2.filter2D(Ixy, -1, window)

    det = (Ixx * Iyy) - (Ixy * Ixy)
    trace = Ixx + Iyy
    # Compute corner response (Harris score)
    corner_response = det - (k * (trace ** 2))

    # corner_response -= np.min(corner_response)
    # corner_response /= np.max(corner_response)

    # Apply non-maximum suppression using numpy vectorized operations
    # Create a response map with padding to make neighborhood operations easier
    offset = window_size // 2
    response_padded = np.pad(corner_response, offset, mode='constant')

    # Create a mask for threshold
    threshold_mask = corner_response > threshold

    # For each pixel, create a boolean mask where it's a local maximum
    local_max = np.ones_like(corner_response, dtype=bool)

    for dy in range(window_size):
        for dx in range(window_size):
            shifted = response_padded[dy:dy+height, dx:dx+width]
            local_max &= (corner_response >= shifted)

    # Combine the threshold mask and local maximum mask
    corner_mask = threshold_mask & local_max

    # Get the coordinates of the corners
    corner_indices = np.argwhere(corner_mask)
    # Convert to (x, y) format from (row, col)
    corners = [(int(x), int(y)) for y, x in corner_indices]

    return corners


def get_msop_features(image, depth=3):
    """
    Extract Multi-Scale Oriented Patches (MSOP) descriptors

    Args:
        image: Input image
        corners: List of corner coordinates (x, y)
        patch_size: Size of the patch to extract (should be odd)

    Returns:
        descriptors: List of MSOP descriptors for each corner
    """

    h, w = image.shape[:2]

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)

    pyramid = [image]
    # Create Gaussian pyramid
    for i in range(depth - 1):
        P_i = cv2.GaussianBlur(pyramid[-1], (-1, -1), 1)
        P_i = cv2.resize(P_i, (P_i.shape[1] // 2, P_i.shape[0] // 2))
        pyramid.append(P_i)

    interesting_points = []
    for i, p in enumerate(pyramid):
        # Calculate image gradients using Sobel
        # Define Sobel kernels
        sobel_x = np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        delta_P_l_x = cv2.filter2D(p, -1, sobel_x)
        delta_P_l_y = cv2.filter2D(p, -1, sobel_y)
        delta_P_l_x = cv2.GaussianBlur(delta_P_l_x, (-1, -1), 1)
        delta_P_l_y = cv2.GaussianBlur(delta_P_l_y, (-1, -1), 1)

        # compute outer product of gradients
        h = p.shape[0]
        w = p.shape[1]
        outer_product = np.zeros((h, w, 2, 2), dtype=np.float32)
        outer_product[:, :, 0, 0] = delta_P_l_x * delta_P_l_x
        outer_product[:, :, 0, 1] = delta_P_l_x * delta_P_l_y
        outer_product[:, :, 1, 0] = delta_P_l_x * delta_P_l_y
        outer_product[:, :, 1, 1] = delta_P_l_y * delta_P_l_y

        outer_product[:, :, 0, 0] = cv2.GaussianBlur(
            outer_product[:, :, 0, 0], (-1, -1), 1.5)
        outer_product[:, :, 0, 1] = cv2.GaussianBlur(
            outer_product[:, :, 0, 1], (-1, -1), 1.5)
        outer_product[:, :, 1, 0] = cv2.GaussianBlur(
            outer_product[:, :, 1, 0], (-1, -1), 1.5)
        outer_product[:, :, 1, 1] = cv2.GaussianBlur(
            outer_product[:, :, 1, 1], (-1, -1), 1.5)

        det = outer_product[:, :, 0, 0] * outer_product[:, :, 1, 1] - \
            outer_product[:, :, 0, 1] * outer_product[:, :, 1, 0]
        trace = outer_product[:, :, 0, 0] + outer_product[:, :, 1, 1]

        f_HM = det / (trace + 1e-6)

        threshold_mask = f_HM > 10

        padded_f_HM = np.pad(f_HM, ((1, 1), (1, 1)), mode='constant')
        local_max = np.ones_like(f_HM, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                shifted = padded_f_HM[dy:dy + h, dx:dx + w]
                local_max &= (f_HM >= shifted)
        final_mask = threshold_mask & local_max

        corners = np.argwhere(final_mask)

        # find_theta
        delta_P_lo_x = cv2.filter2D(p, -1, sobel_x)
        delta_P_lo_y = cv2.filter2D(p, -1, sobel_y)
        delta_P_lo_x = cv2.GaussianBlur(delta_P_lo_x, (-1, -1), 4.5)
        delta_P_lo_y = cv2.GaussianBlur(delta_P_lo_y, (-1, -1), 4.5)

        thetas = np.arctan2(delta_P_lo_y, delta_P_lo_x)
        theta = thetas[corners[:, 0], corners[:, 1]]
        scores = f_HM[corners[:, 0], corners[:, 1]]
        depths = np.ones(corners.shape[0], dtype=np.float32) * i

        cur_points = np.array(
            [corners[:, 1], corners[:, 0], theta, scores, depths]).T

        # make all interesting points into a numpy array of [x, y, theta, score]
        interesting_points.append(cur_points)

    interesting_points = np.concatenate(interesting_points, axis=0)

    return interesting_points


def adaptive_non_maximal_suppression(interesting_points, maximum_points):
    """
    Perform adaptive non-maximal suppression on the scores

    Args:
        interesting_points: Array of interesting points
        maximum_points: Maximum number of points to keep

    Returns:
        suppressed_scores: Array of suppressed scores
    """
    n = interesting_points.shape[0]
    if n < maximum_points:
        return interesting_points

    min_distance = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x1, y1 = interesting_points[i, :2]
        score = interesting_points[i, 3]
        mask = interesting_points[:, 3] * 0.9 > score
        mask[i] = False
        x2 = interesting_points[mask, 0]
        y2 = interesting_points[mask, 1]
        distances = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        min_distance[i] = np.min(distances) if distances.size > 0 else 0

    sorted_indices = np.argsort(min_distance)[::-1]

    suppressed_scores = interesting_points[sorted_indices[:maximum_points]]

    return np.array(suppressed_scores)


def get_msop_descriptors(image, points, patch_size=8, spacing=2):

    n = points.shape[0]
    m = patch_size * patch_size

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image)

    patch_indices = []
    # create a patch index [[0,0], [0,1], [0,2], ... , [1,0], ..., [patch_size-1, patch_size-1]]
    patch_indices = np.array(
        [[i, j] for i in range(patch_size) for j in range(patch_size)], dtype=np.float32)
    patch_indices = patch_indices - (patch_size - 1) / 2

    sizes = points[:, 4] + 1

    theta = points[:, 2]
    rotation_matrices = np.zeros((points.shape[0], 2, 2), dtype=np.float32)
    rotation_matrices[:, 0, 0] = np.cos(theta)
    rotation_matrices[:, 0, 1] = -np.sin(theta)
    rotation_matrices[:, 1, 0] = np.sin(theta)
    rotation_matrices[:, 1, 1] = np.cos(theta)

    patch_indices = rotation_matrices[:, np.newaxis,
                                      :, :] @ patch_indices[np.newaxis, :, :, np.newaxis]
    patch_indices = patch_indices.reshape(n, m, 2)

    patch_indices *= sizes[:, np.newaxis, np.newaxis]

    patch_indices *= spacing

    patch_indices = points[:, np.newaxis, :2] + patch_indices

    patch_indices = patch_indices.astype(np.int32)

    patch_indices[:, :, 0] = np.clip(
        patch_indices[:, :, 0], 0, image.shape[1] - 1)
    patch_indices[:, :, 1] = np.clip(
        patch_indices[:, :, 1], 0, image.shape[0] - 1)

    features = image[patch_indices[:, :, 1], patch_indices[:, :, 0]]

    return points, features


def draw_points(image_file, image, corners, output_folder):
    """
    Draw corners on the image and save to output folder

    Args:
        image: Input image
        corners: List of corner coordinates (x, y)
        output_folder: Folder to save the output images

    Returns:
        None
    """
    # Convert to color if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw corners
    for c in corners:
        x = int(c[0])
        y = int(c[1])
        theta = c[2]
        size = int(c[4] * 8 + 8) * 2

        rect = cv2.boxPoints(
            ((x, y), (size, size), theta * 180 / np.pi)).astype(int)
        # Draw the rectangle
        cv2.polylines(image, [rect], isClosed=True,
                      color=(0, 0, 255), thickness=1)

    # Save the image with corners drawn
    output_path = os.path.join(output_folder, "corners_" + image_file)
    cv2.imwrite(output_path, image)


def detect_features(image_file, image, output_folder, max_points=250, draw=False):
    """
    Detect features in an image using Harris corner detector and MSOP descriptors

    Args:
        image: Input image
        output_folder: Folder to save the output images

    Returns:
        features: Dictionary containing keypoints and descriptors
    """

    keypoints = get_msop_features(image)

    keypoints = adaptive_non_maximal_suppression(keypoints, max_points)

    keypoints, features = get_msop_descriptors(image, keypoints)

    if draw:
        # Draw corners on the image and save to output folder
        draw_points(image_file, image, keypoints, output_folder)

    # Return features
    return {
        'keypoints': keypoints[:, :2],
        'descriptors': features
    }
