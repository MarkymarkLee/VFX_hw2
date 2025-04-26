import numpy as np


def colored_print(text, color):
    """
    Print text in a specific color
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['reset'])}{text}{colors['reset']}")


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
    b = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]

        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        b.append([u])
        b.append([v])

    A = np.array(A)
    b = np.array(b)

    # Solve for h
    new_A = A.T @ A
    new_b = A.T @ b
    h = np.linalg.solve(new_A, new_b)
    h = h.reshape(2, 3)
    h = np.append(h, np.array([0, 0, 1]))

    return h.reshape(3, 3)


def calc_inliers(H, src_points, dst_points, distance_threshold):
    """
    Calculate inliers based on the homography matrix

    Args:
        H: Homography matrix
        src_points: List of source points [(x1, y1), (x2, y2), ...]
        dst_points: List of destination points [(x1, y1), (x2, y2), ...]

    Returns:
        inliers: List of inlier indices
    """
    # Convert to homogeneous coordinates
    src_homogeneous = np.hstack([src_points, np.ones((len(src_points), 1))])

    # Apply homography to all points at once
    transformed = np.dot(H, src_homogeneous.T).T

    # Convert back from homogeneous coordinates
    transformed_normalized = transformed / \
        (transformed[:, 2:] + 1e-10)  # Avoid division by zero
    transformed_points = transformed_normalized[:, :2]

    # Calculate distances for all points at once
    distances = np.sqrt(
        np.sum((dst_points - transformed_points)**2, axis=1))

    # Get indices of inliers
    inliers = np.where(distances < distance_threshold)[0].tolist()

    return inliers


def ransac_homography(matches, max_iterations=5000, distance_threshold=2.0):
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

        inlier_indices = calc_inliers(
            H, np.array(src_points), np.array(dst_points), distance_threshold)

        # Update best model if more inliers found
        if len(inlier_indices) > max_inliers:
            max_inliers = len(inlier_indices)
            best_H = H
            best_inlier_indices = inlier_indices

    # If no good model found
    if best_H is None:
        return None, []

    if len(best_inlier_indices) < 4:
        return None, []

    # Refine homography using all inliers
    inlier_src = [src_points[i] for i in best_inlier_indices]
    inlier_dst = [dst_points[i] for i in best_inlier_indices]

    # Re-estimate homography with all inliers
    refined_H = estimate_homography(inlier_src, inlier_dst)

    # Get inlier matches
    inlier_indices = calc_inliers(
        refined_H, np.array(src_points), np.array(dst_points), distance_threshold)

    inliers = [matches[i] for i in inlier_indices]

    return refined_H, inliers


def filter_pairs_by_matches(feature_matches, max_matches_per_image=6):
    # Collect all matches for each image
    image_matches = {}
    for pair, matches in feature_matches.items():
        img1_name, img2_name = pair

        # Add match count for img1 -> img2
        if img1_name not in image_matches:
            image_matches[img1_name] = {}
        image_matches[img1_name][img2_name] = len(matches)

        # Add match count for img2 -> img1
        if img2_name not in image_matches:
            image_matches[img2_name] = {}
        image_matches[img2_name][img1_name] = len(matches)

    # Filter to keep only top N matches for each image
    filtered_pairs = set()
    for img_name, matches in image_matches.items():
        # Sort matches by count (descending)
        sorted_matches = sorted(
            matches.items(), key=lambda x: x[1], reverse=True)

        # Take only top N matches
        top_matches = sorted_matches[:max_matches_per_image]

        # Add to filtered pairs
        for match_img, _ in top_matches:
            if img_name < match_img:  # Ensure each pair is added only once
                filtered_pairs.add((img_name, match_img))
            else:
                filtered_pairs.add((match_img, img_name))

    return filtered_pairs


def match_images(feature_matches, max_matches_per_image=3):
    """
    Find consistent image matches using RANSAC, limiting each image 
    to match with only the top N images that have the most matches with it

    Args:
        feature_matches: Dictionary of matched features between image pairs
        max_matches_per_image: Maximum number of matches per image (default: 6)

    Returns:
        consistent_matches: Dictionary with image pairs as keys and 
                          (homography, inlier_matches) as values
    """
    consistent_matches = {}

    filtered_pairs = filter_pairs_by_matches(
        feature_matches, max_matches_per_image)

    # Process only the filtered pairs
    for pair in filtered_pairs:
        if pair in feature_matches:
            img1_name, img2_name = pair
            matches = feature_matches[pair]
            inlier_threshold = 5 + 0.22 * len(matches)

            # Find homography using RANSAC
            H, inliers = ransac_homography(matches)
            inliers_count = len(inliers)

            # If a good homography was not found
            if H is None or len(inliers) < 4:
                continue

            if inliers_count >= inlier_threshold:
                # Store the homography and inliers
                consistent_matches[pair] = (H, inliers)
                colored_print(
                    f"ACCEPTED: {inliers_count}/{len(matches)} matches for {img1_name} and {img2_name}",
                    color='green')
            else:
                colored_print(
                    f"REJECTED: {inliers_count}/{len(matches)} matches for {img1_name} and {img2_name}",
                    color='red')

    return consistent_matches
