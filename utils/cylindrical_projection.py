import cv2
import numpy as np


def _deprecated_cylindrical_project(image_path, focal_length):
    """
    Project an image onto a cylindrical surface

    Args:
        image_path: Path to the input image
        focal_length: Focal length of the camera

    Returns:
        warped: Image warped onto cylindrical surface
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Get image dimensions
    h, w = image.shape[:2]

    # Create output image with same size
    warped = np.zeros_like(image)

    # Get center of image
    center_x = w / 2
    center_y = h / 2

    # Map each pixel to the cylinder
    for y in range(h):
        for x in range(w):
            # Convert to cylindrical coordinates
            # theta = (x - center_x) / focal_length
            # h_prime = (y - center_y) / focal_length

            # Convert back to image coordinates
            x_source = focal_length * \
                np.tan((x - center_x) / focal_length) + center_x
            y_source = (y - center_y) / \
                np.cos((x - center_x) / focal_length) + center_y

            # Check if source coordinates are within image bounds
            if 0 <= x_source < w and 0 <= y_source < h:
                # Use bilinear interpolation for smoother results
                x1, y1 = int(x_source), int(y_source)
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

                # Calculate interpolation weights
                dx = x_source - x1
                dy = y_source - y1

                # Perform bilinear interpolation
                pixel = (1 - dx) * (1 - dy) * image[y1, x1] + \
                    dx * (1 - dy) * image[y1, x2] + \
                    (1 - dx) * dy * image[y2, x1] + \
                    dx * dy * image[y2, x2]

                warped[y, x] = pixel

    # Optimize: The above implementation is very slow due to the nested loops
    # For production code, you would vectorize this calculation

    return warped


def optimized_cylindrical_project(image_path, focal_length):
    """
    Project an image onto a cylindrical surface (optimized version)

    Args:
        image_path: Path to the input image
        focal_length: Focal length of the camera

    Returns:
        warped: Image warped onto cylindrical surface
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Get image dimensions
    h, w = image.shape[:2]

    # Create meshgrid for coordinates
    x_range = np.arange(w)
    y_range = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Get center of image
    center_x = w / 2
    center_y = h / 2

    # Convert to cylindrical coordinates
    theta = (x_grid - center_x) / focal_length
    h_prime = (y_grid - center_y) / focal_length

    # Convert back to image coordinates
    x_source = focal_length * np.tan(theta) + center_x
    y_source = h_prime * np.cos(theta) + center_y

    # Create map for remap function
    map_x = x_source.astype(np.float32)
    map_y = y_source.astype(np.float32)

    # Remap the image
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return warped


def cylindrical_project(image_path, focal_length):
    """
    Main function to project an image onto a cylindrical surface

    Args:
        image_path: Path to the input image
        focal_length: Focal length of the camera

    Returns:
        warped: Image warped onto cylindrical surface
    """
    # Use the optimized version for better performance
    return optimized_cylindrical_project(image_path, focal_length)
