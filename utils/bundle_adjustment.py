import numpy as np
from scipy.optimize import least_squares


def adjust_bundle(consistent_matches):
    """
    Perform bundle adjustment to optimize camera parameters

    Args:
        consistent_matches: Dictionary with image pairs as keys and 
                          (homography, inlier_matches) as values

    Returns:
        camera_params: Dictionary of optimized camera parameters for each image
    """
    # If no consistent matches, return empty dict
    if not consistent_matches:
        return {}

    # Collect all unique images
    unique_images = set()
    for img1, img2 in consistent_matches.keys():
        unique_images.add(img1)
        unique_images.add(img2)

    # Create a graph of image connections
    graph = {img: [] for img in unique_images}
    for (img1, img2), (H, _) in consistent_matches.items():
        graph[img1].append((img2, H))
        # Add inverse homography for the other direction
        H_inv = np.linalg.inv(H)
        graph[img2].append((img1, H_inv))

    # Find the image with the most connections as the reference image
    reference_image = max(graph.keys(), key=lambda img: len(graph[img]))

    # Initialize camera parameters with identity for reference image
    camera_params = {reference_image: np.eye(3)}
    visited = {reference_image}

    # Use breadth-first search to propagate camera parameters
    queue = [reference_image]
    while queue:
        current_img = queue.pop(0)
        current_H = camera_params[current_img]

        for neighbor_img, H in graph[current_img]:
            if neighbor_img not in visited:
                # Compute the homography from reference to neighbor
                neighbor_H = np.dot(current_H, H)
                camera_params[neighbor_img] = neighbor_H
                visited.add(neighbor_img)
                queue.append(neighbor_img)

    # Define error function for optimization
    def reprojection_error(params, n_images, matches_data):
        """
        Calculate reprojection error for bundle adjustment

        Args:
            params: Flattened camera parameters
            n_images: Number of images
            matches_data: List of (img1_idx, img2_idx, [(x1, y1), (x2, y2), ...])

        Returns:
            errors: Flattened reprojection errors
        """
        # Reshape params into camera matrices (identity for reference)
        camera_matrices = []
        param_idx = 0
        for i in range(n_images):
            if i == 0:  # Reference image
                camera_matrices.append(np.eye(3))
            else:
                # Extract 8 parameters (H has 8 DoF with H[2,2]=1)
                h = np.ones(9)
                h[:8] = params[param_idx:param_idx+8]
                camera_matrices.append(h.reshape(3, 3))
                param_idx += 8

        # Calculate reprojection errors
        all_errors = []
        for img1_idx, img2_idx, points in matches_data:
            H1 = camera_matrices[img1_idx]
            H2 = camera_matrices[img2_idx]

            # Combined homography from img1 to img2
            H = np.dot(np.linalg.inv(H2), H1)

            for (x1, y1), (x2, y2) in points:
                # Transform point from img1 to img2
                p1 = np.array([x1, y1, 1])
                p1_transformed = np.dot(H, p1)
                p1_transformed /= p1_transformed[2]  # Normalize
                tx, ty = p1_transformed[0], p1_transformed[1]

                # Calculate error
                err_x = x2 - tx
                err_y = y2 - ty
                all_errors.extend([err_x, err_y])

        return all_errors

    # Extract matches data for optimization
    image_to_idx = {img: i for i, img in enumerate(unique_images)}
    matches_data = []
    for (img1, img2), (_, inliers) in consistent_matches.items():
        img1_idx = image_to_idx[img1]
        img2_idx = image_to_idx[img2]
        matches_data.append((img1_idx, img2_idx, inliers))

    # Flatten initial camera parameters (exclude reference)
    initial_params = []
    for img in unique_images:
        if img != reference_image:
            H = camera_params[img]
            # Flatten and exclude last element (H[2,2] = 1)
            h_flat = H.flatten()
            initial_params.extend(h_flat[:8])

    # Optimize camera parameters
    n_images = len(unique_images)
    result = least_squares(reprojection_error, initial_params,
                           args=(n_images, matches_data),
                           method='lm', verbose=2)

    # Extract optimized parameters
    optimized_params = {}
    optimized_params[reference_image] = np.eye(3)

    param_idx = 0
    for img in unique_images:
        if img != reference_image:
            h = np.ones(9)
            h[:8] = result.x[param_idx:param_idx+8]
            optimized_params[img] = h.reshape(3, 3)
            param_idx += 8

    return optimized_params
