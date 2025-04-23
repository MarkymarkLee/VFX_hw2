import cv2
import numpy as np


def cylindrical_project(image, focal_length, radius=None):
    """
    Project an image onto a cylindrical surface (optimized version)

    Args:
        image_path: Path to the input image
        focal_length: Focal length of the camera
        radius: Radius of the cylindrical projection (default is None, which uses focal_length)

    Returns:
        warped: Image warped onto cylindrical surface
    """
    # Get image dimensions
    h, w = image.shape[:2]

    if radius is None:
        radius = focal_length

    # Create meshgrid for coordinates
    x_range = np.arange(w)
    y_range = np.arange(h)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Get center of image
    center_x = w / 2
    center_y = h / 2

    # Convert to cylindrical coordinates
    theta = np.arctan((x_grid - center_x) / focal_length)
    h = (y_grid - center_y) / np.sqrt(
        (x_grid - center_x) ** 2 + focal_length ** 2)

    # Convert back to image coordinates
    x_source = radius * theta + center_x
    y_source = radius * h + center_y

    # Create map for remap function
    map_x = x_source.astype(np.float32)
    map_y = y_source.astype(np.float32)

    # Remap the image
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)

    return warped
