import cv2
import numpy as np


def cylindrical_project_points(points, center, focal_length, radius=None):
    """
    Project 2D points onto a cylindrical surface

    Args:
        points: 2D points to be projected
        center: Center of the cylindrical projection
        focal_length: Focal length of the camera
        radius: Radius of the cylindrical projection (default is None, which uses focal_length)

    Returns:
        projected_points: 2D points on the cylindrical surface
        overflow: Boolean array
    """
    if radius is None:
        radius = focal_length

    shifted_points = points - center

    # Convert to cylindrical coordinates
    theta = radius * np.arctan(shifted_points[:, 0] / focal_length)
    h = shifted_points[:, 1] * radius / \
        np.sqrt(focal_length**2 + shifted_points[:, 0]**2)

    return np.column_stack((theta, h))


def project_to_canvas(image, center, focal_length, homography, translation, canvas_size, radius=None):
    """
    Project an image onto a cylindrical surface (optimized version)

    Args:
        image: Input image to be projected
        center: Center of the cylindrical projection
        focal_length: Focal length of the camera
        homography: Homography matrix for the image
        translation: Translation vector for the image
        canvas_size: Size of the canvas for the output image (width, height)
        radius: Radius of the cylindrical projection (default is None, which uses focal_length)

    Returns:
        warped: Image warped onto cylindrical surface
    """

    h, w = image.shape[:2]

    if radius is None:
        radius = focal_length

    canvas_w, canvas_h = canvas_size
    new_x = np.arange(canvas_w)
    new_y = np.arange(canvas_h)
    new_x, new_y = np.meshgrid(new_x, new_y)
    new_x = new_x.astype(np.float32)
    new_y = new_y.astype(np.float32)

    new_x -= translation[0]
    new_y -= translation[1]

    # new_x -= center[0]
    # new_y -= center[1]

    new_x = np.tan(new_x / radius) * focal_length
    new_y = new_y / radius * np.sqrt(focal_length**2 + new_x**2)

    new_x = new_x + center[0]
    new_y = new_y + center[1]

    homography_inv = np.linalg.inv(homography)
    new_points = np.column_stack(
        (new_x.ravel(), new_y.ravel(), np.ones(new_x.size)))
    new_points = np.dot(homography_inv, new_points.T).T
    new_points /= new_points[:, 2:]
    new_points = new_points[:, :2]
    new_points = new_points.reshape(canvas_h, canvas_w, 2)

    new_x = new_points[:, :, 0].astype(np.float32)
    new_y = new_points[:, :, 1].astype(np.float32)

    mask = (0 <= new_x) & (new_x < w) & (0 <= new_y) & (new_y < h)
    new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
    new_y = np.clip(new_y, 0, h - 1).astype(np.float32)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas = cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR,
                       dst=canvas, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    canvas[~mask] = 0
    return canvas, mask
