import numpy as np
import cv2


def create_panorama_canvas(images, camera_params):
    """
    Create a canvas for the panorama and place images on it

    Args:
        images: List of (image_name, image_data)
        camera_params: Dictionary of camera parameters for each image

    Returns:
        canvas: Empty canvas for the panorama
        corners: Dictionary of corner coordinates for each image
    """
    # Find min and max coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    corners = {}

    for img_name, img in images:
        # Get image dimensions
        h, w = img.shape[:2]

        # Get homography for this image
        H = camera_params.get(img_name, np.eye(3))

        # Calculate corners of the image in the panorama space
        corners_before = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ])

        # Transform corners
        corners_after = []
        for corner in corners_before:
            transformed = np.dot(H, corner)
            transformed /= transformed[2]  # Normalize
            corners_after.append(transformed[:2])

        corners[img_name] = np.array(corners_after)

        # Update min and max coordinates
        for x, y in corners_after:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Calculate canvas dimensions
    width = int(np.ceil(max_x - min_x))
    height = int(np.ceil(max_y - min_y))

    # Shift corners to account for the offset
    for img_name in corners:
        corners[img_name] -= np.array([min_x, min_y])

    # Create empty canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    return canvas, corners


def simple_blend(images, camera_params):
    """
    Simple approach to place images on a panorama canvas

    Args:
        images: List of (image_name, image_data)
        camera_params: Dictionary of camera parameters for each image

    Returns:
        panorama: Blended panorama image
    """
    # Create canvas
    canvas, corners = create_panorama_canvas(images, camera_params)
    height, width = canvas.shape[:2]

    # Create weight map for blending
    weight_map = np.zeros((height, width), dtype=np.float32)

    # Place each image on the canvas
    for img_name, img in images:
        # Get homography for this image
        H = camera_params.get(img_name, np.eye(3))

        # Create mask
        mask = np.ones(img.shape[:2], dtype=np.float32)

        # Feather the edges of the mask
        r = 50  # Feather radius
        mask[:r, :] = np.linspace(0, 1, r)[:, np.newaxis]
        mask[-r:, :] = np.linspace(1, 0, r)[:, np.newaxis]
        mask[:, :r] = np.minimum(
            mask[:, :r], np.linspace(0, 1, r)[np.newaxis, :])
        mask[:, -r:] = np.minimum(mask[:, -r:],
                                  np.linspace(1, 0, r)[np.newaxis, :])

        # Warp image and mask
        warped_img = cv2.warpPerspective(img, H, (width, height))
        warped_mask = cv2.warpPerspective(mask, H, (width, height))

        # Update weight map
        weight_map = np.maximum(weight_map, warped_mask)

        # Blend images
        for c in range(3):
            canvas[:, :, c] = np.where(warped_mask > 0,
                                       canvas[:, :, c] * (1 - warped_mask) +
                                       warped_img[:, :, c] * warped_mask,
                                       canvas[:, :, c])

    return canvas


def poisson_blend(images, camera_params):
    """
    Poisson blending for seamless image stitching

    Args:
        images: List of (image_name, image_data)
        camera_params: Dictionary of camera parameters for each image

    Returns:
        panorama: Blended panorama image
    """
    # Start with simple blending as initial panorama
    panorama = simple_blend(images, camera_params)

    # Using the result of simple blending is a simplification
    # A full Poisson blending implementation would be more complex
    # and would involve solving Poisson equations to minimize
    # the gradient differences at the boundaries

    return panorama


def blend_images(images, camera_params):
    """
    Blend images to create a panorama

    Args:
        images: List of (image_name, image_data)
        camera_params: Dictionary of camera parameters for each image

    Returns:
        panorama: Blended panorama image
    """
    # Use poisson blending for seamless stitching
    return poisson_blend(images, camera_params)
