import cv2
import numpy as np
from PIL import Image


def find_largest_rectangle(mask):
    """
    Find the largest rectangle within a panorama mask

    Args:
        mask: Binary mask of the panorama (non-zero where there's content)

    Returns:
        rect: (x, y, width, height) of the largest rectangle
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return the whole image
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0])

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h)


def rectangle_panorama(panorama):
    """
    Crop the panorama to a rectangular shape, removing irregular boundaries

    Args:
        panorama: Input panorama image

    Returns:
        rectangled_panorama: Panorama cropped to a rectangular shape
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(panorama, Image.Image):
        panorama = np.array(panorama)

    # Create a mask where non-zero pixels are in the panorama
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the largest rectangle in the mask
    x, y, w, h = find_largest_rectangle(mask)

    # Crop the panorama to this rectangle
    rectangled = panorama[y:y+h, x:x+w]

    # Convert back to PIL Image for consistency
    return Image.fromarray(rectangled)
