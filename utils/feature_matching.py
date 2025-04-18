import numpy as np


def match_features(features_dict, ratio_threshold=0.8):
    """
    Match features between image pairs using ratio test

    Args:
        features_dict: Dictionary with image filenames as keys and feature dictionaries as values
        ratio_threshold: Threshold for Lowe's ratio test

    Returns:
        matches: Dictionary with image pairs as keys and matched keypoints as values
    """
    # Initialize matches dictionary
    matches = {}

    # Get list of image names
    image_names = list(features_dict.keys())

    # Compare each pair of images
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            img1_name = image_names[i]
            img2_name = image_names[j]

            # Get features for both images
            img1_features = features_dict[img1_name]
            img2_features = features_dict[img2_name]

            # Get keypoints and descriptors
            kp1 = img1_features['keypoints']
            desc1 = img1_features['descriptors']
            kp2 = img2_features['keypoints']
            desc2 = img2_features['descriptors']

            # Convert lists to numpy arrays for faster computation
            if not isinstance(desc1, np.ndarray):
                desc1 = np.array(desc1)
            if not isinstance(desc2, np.ndarray):
                desc2 = np.array(desc2)

            # Match descriptors using nearest neighbor and ratio test
            matched_pairs = []

            # For each descriptor in first image
            for idx1, desc in enumerate(desc1):
                # Calculate Euclidean distances to all descriptors in second image
                distances = np.sqrt(np.sum((desc2 - desc) ** 2, axis=1))

                # Sort distances and get indices of two closest matches
                sorted_idx = np.argsort(distances)

                # Apply Lowe's ratio test (if second best match is significantly worse than first)
                if len(sorted_idx) >= 2:
                    best_idx = sorted_idx[0]
                    second_best_idx = sorted_idx[1]

                    # If best match is significantly better than second best
                    if distances[best_idx] < ratio_threshold * distances[second_best_idx]:
                        matched_pairs.append((kp1[idx1], kp2[best_idx]))

            # Store matches if any were found
            if matched_pairs:
                pair_key = (img1_name, img2_name)
                matches[pair_key] = matched_pairs

    return matches
