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

    shifted_x = points[:, 0] - center[0]
    shifted_y = points[:, 1] / points[:, 2] - center[1]
    overflow = points[:, 2] == -1

    # Convert to cylindrical coordinates
    theta = radius * np.arctan(shifted_x / focal_length)
    shifted_x_sign = np.sign(shifted_x)
    theta[overflow] += shifted_x_sign[overflow] * radius * np.pi / 2

    h = shifted_y * radius / \
        np.sqrt(focal_length**2 + shifted_x**2)
    h[overflow] *= -1

    theta += center[0]
    h += center[1]

    return np.column_stack((theta, h))


def cylindrical_project(image, focal_length, radius=None):
    """
    Project an image onto a cylindrical surface

    Args:
        image: Input image to be projected
        focal_length: Focal length of the camera
        radius: Radius of the cylindrical projection (default is None, which uses focal_length)

    Returns:
        warped: Image warped onto cylindrical surface
    """

    h, w = image.shape[:2]

    if radius is None:
        radius = focal_length

    center = (w / 2, h / 2)

    # Create a grid of points
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    x -= center[0]
    y -= center[1]

    # Project the points onto the cylindrical surface
    x = np.tan(x / radius) * focal_length
    y = y / radius * np.sqrt(focal_length**2 + x**2)
    # y = y / np.cos(x / focal_length)

    x = x + center[0]
    y = y + center[1]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR,
                       dst=canvas, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask = cv2.remap(np.ones((h, w), dtype=np.float32), x, y,
                     interpolation=cv2.INTER_LINEAR,
                     dst=np.zeros((h, w), dtype=np.float32), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # calculate a bounding box for the mask
    contours = cv2.findContours(mask.astype(
        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) == 0:
        return canvas, mask == 1
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    canvas = canvas[y:y + h, x:x + w]
    mask = mask[y:y + h, x:x + w]

    canvas[mask != 1] = 0

    return canvas, mask == 1


def project_to_canvas(image, image_mask, homography, translation, canvas_size):
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

    canvas_w, canvas_h = canvas_size
    new_x = np.arange(canvas_w)
    new_y = np.arange(canvas_h)
    new_x, new_y = np.meshgrid(new_x, new_y)
    new_x = new_x.astype(np.float32)
    new_y = new_y.astype(np.float32)

    new_x -= translation[0]
    new_y -= translation[1]

    homography_inv = np.linalg.inv(homography)
    new_points = np.column_stack(
        (new_x.ravel(), new_y.ravel(), np.ones(canvas_w * canvas_h)))
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
    image_mask = image_mask.astype(np.float32)
    image_mask = cv2.remap(image_mask, new_x, new_y, interpolation=cv2.INTER_LINEAR,
                           dst=np.zeros((canvas_h, canvas_w),
                                        dtype=np.float32),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    image_mask[mask != 1] = 0
    canvas[image_mask != 1] = 0

    return canvas, image_mask == 1
