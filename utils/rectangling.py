import cv2
import numpy as np


def find_fitting_curve(points):
    """
    Find the best fitting curve for the given points using polynomial regression.

    Args:
        points (numpy.ndarray): The input points.

    Returns:
        fitted_curve (numpy.ndarray): The new points with the same x.
    """
    x = np.arange(len(points))
    y = points
    z = np.polyfit(x, y, 5)
    new_points = np.polyval(z, x)
    return new_points


def rectangle_panorama(panorama, panorama_mask):
    """
    Rectangles the panorama image using the provided mask.

    Args:
        panorama (numpy.ndarray): The input panorama image.
        panorama_mask (numpy.ndarray): The mask for the panorama image.

    Returns:
        numpy.ndarray: The rectangled panorama image.
    """

    # Check if the panorama is in landscape mode
    w, h = panorama.shape[1], panorama.shape[0]
    if w < h:
        panorama = cv2.rotate(panorama, cv2.ROTATE_90_CLOCKWISE)
        panorama_mask = cv2.rotate(panorama_mask*255, cv2.ROTATE_90_CLOCKWISE)
        panorama_mask = panorama_mask > 0

    w, h = panorama.shape[1], panorama.shape[0]
    centers = panorama_mask.T @ np.arange(h) / h
    centers = find_fitting_curve(centers)

    h_count = np.sum(panorama_mask, axis=0)

    new_w = w
    new_h = h

    x = np.arange(new_w)
    y = np.arange(new_h)
    x, y = np.meshgrid(x, y)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    y += centers[np.newaxis, :] - new_h / 2

    transformed_panorama = cv2.remap(
        panorama, x, y, interpolation=cv2.INTER_LINEAR)
    transformed_mask = cv2.remap(
        panorama_mask.astype(np.uint8), x, y, interpolation=cv2.INTER_NEAREST)
    transformed_mask = transformed_mask > 0

    good_row = np.sum(transformed_mask, axis=1) >= w * 0.8

    rectangle = transformed_panorama[good_row]
    rectangle_mask = transformed_mask[good_row]

    holes = (rectangle_mask == 0).astype(np.uint8)

    final_panorama = cv2.inpaint(
        rectangle, holes, 10,  cv2.INPAINT_TELEA)

    return final_panorama
