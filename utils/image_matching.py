import numpy as np


def estimate_homography(src_points, dst_points):
    """
    Estimate homography matrix using Direct Linear Transform (DLT)

    Args:
        src_points: List of source points [(x1, y1), (x2, y2), ...]
        dst_points: List of destination points [(x1, y1), (x2, y2), ...]

    Returns:
        H: 3x3 homography matrix
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError(
            "At least 4 points are required to estimate homography")

    # Create matrix A for DLT
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]

        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    A = np.array(A)

    # Solve for h (SVD)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]

    # Reshape to 3x3 matrix
    H = h.reshape(3, 3)

    # Normalize
    H = H / H[2, 2]

    return H


def ransac_homography(matches, max_iterations=1000, distance_threshold=5.0):
    """
    Use RANSAC to find the best homography matrix

    Args:
        matches: List of matched keypoint pairs [((x1, y1), (x2, y2)), ...]
        max_iterations: Maximum number of RANSAC iterations
        distance_threshold: Threshold for considering a point as inlier

    Returns:
        best_H: Best homography matrix
        inliers: List of inlier matches
    """
    if len(matches) < 4:
        return None, []

    # Extract points from matches
    src_points = [match[0] for match in matches]
    dst_points = [match[1] for match in matches]

    best_H = None
    max_inliers = 0
    best_inlier_indices = []

    num_matches = len(matches)

    for _ in range(max_iterations):
        # Randomly select 4 matches
        random_indices = np.random.choice(num_matches, 4, replace=False)

        # Get points for selected matches
        sample_src = [src_points[i] for i in random_indices]
        sample_dst = [dst_points[i] for i in random_indices]

        # Estimate homography
        try:
            H = estimate_homography(sample_src, sample_dst)
        except np.linalg.LinAlgError:
            # Singular matrix, skip this iteration
            continue

        # Count inliers
        inlier_indices = []
        for i in range(num_matches):
            x, y = src_points[i]
            u, v = dst_points[i]

            # Convert to homogeneous coordinates
            src_homogeneous = np.array([x, y, 1])

            # Apply homography
            transformed = np.dot(H, src_homogeneous)

            # Convert back from homogeneous coordinates
            transformed /= transformed[2]
            tx, ty = transformed[0], transformed[1]

            # Calculate distance
            distance = np.sqrt((u - tx)**2 + (v - ty)**2)

            # Check if it's an inlier
            if distance < distance_threshold:
                inlier_indices.append(i)

        # Update best model if more inliers found
        if len(inlier_indices) > max_inliers:
            max_inliers = len(inlier_indices)
            best_H = H
            best_inlier_indices = inlier_indices

    # If no good model found
    if best_H is None:
        return None, []

    # Refine homography using all inliers
    inlier_src = [src_points[i] for i in best_inlier_indices]
    inlier_dst = [dst_points[i] for i in best_inlier_indices]

    # Re-estimate homography with all inliers
    refined_H = estimate_homography(inlier_src, inlier_dst)

    # Get inlier matches
    inliers = [matches[i] for i in best_inlier_indices]

    return refined_H, inliers


def match_images(feature_matches):
    """
    Find consistent image matches using RANSAC

    Args:
        feature_matches: Dictionary of matched features between image pairs

    Returns:
        consistent_matches: Dictionary with image pairs as keys and 
                          (homography, inlier_matches) as values
    """
    consistent_matches = {}

    # Process each image pair
    for pair, matches in feature_matches.items():
        img1_name, img2_name = pair

        # Find homography using RANSAC
        H, inliers = ransac_homography(matches)

        # If a good homography was found
        if H is not None and len(inliers) >= 4:
            consistent_matches[pair] = (H, inliers)
            print(
                f"Found {len(inliers)} consistent matches between {img1_name} and {img2_name}")

    return consistent_matches
