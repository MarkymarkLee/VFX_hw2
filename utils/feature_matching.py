import numpy as np
from scipy.spatial import KDTree


def match_features(features_dict, ratio_threshold=0.8, cross_check=True, max_distance=0.4):
    """
    Match features between image pairs using ratio test

    Args:
        features_dict: Dictionary with image filenames as keys and feature dictionaries as values, 
                      file dictionaries have 2 keys: "keypoints" and "descriptors"
        ratio_threshold: Threshold for ratio test (0.8 is a common value)
        cross_check: Whether to perform cross-checking
        max_distance: Maximum distance threshold for valid matches

    Returns:
        matches: Dictionary with image pairs as keys and matched keypoints as values
    """
    # Initialize matches dictionary
    matches = {}

    # Get list of image names
    image_names = list(features_dict.keys())

    # For each pair of images
    for i in range(len(image_names)):
        img1 = image_names[i]
        print(f"Matching features for {img1}...", end='\r')
        desc1 = features_dict[img1]['descriptors']
        kp1 = features_dict[img1]['keypoints']

        # Convert to numpy array if needed
        if not isinstance(desc1, np.ndarray):
            desc1 = np.array(desc1)

        # Normalize descriptors for better distance comparison
        norms1 = np.sqrt(np.sum(desc1 * desc1, axis=1))
        norms1[norms1 == 0] = 1  # Avoid division by zero
        desc1_normalized = desc1 / norms1[:, np.newaxis]
        # desc1_normalized = desc1

        for j in range(i + 1, len(image_names)):
            img2 = image_names[j]
            desc2 = features_dict[img2]['descriptors']
            kp2 = features_dict[img2]['keypoints']

            # Convert to numpy array if needed
            if not isinstance(desc2, np.ndarray):
                desc2 = np.array(desc2)

            # Normalize descriptors
            norms2 = np.sqrt(np.sum(desc2 * desc2, axis=1))
            norms2[norms2 == 0] = 1
            desc2_normalized = desc2 / norms2[:, np.newaxis]
            # desc2_normalized = desc2

            # Create KDTree for the second image descriptors
            tree = KDTree(desc2_normalized)

            # Find 4 nearest neighbors for ratio test
            distances, indices = tree.query(desc1_normalized, k=4)

            # Store forward matches (img1 -> img2)
            img1_to_img2 = []

            # Apply ratio test and distance threshold
            for idx, (dist, ind) in enumerate(zip(distances, indices)):
                if len(dist) > 1 and dist[0] < ratio_threshold * dist[1] and dist[0] < max_distance:
                    img1_to_img2.append((idx, ind[0], dist[0]))

            # If cross-checking is enabled
            if cross_check:
                # Build KDTree for the first image descriptors
                tree_reverse = KDTree(desc1_normalized)

                # Find nearest neighbors for second image
                distances_reverse, indices_reverse = tree_reverse.query(
                    desc2_normalized, k=2)

                # Store backward matches (img2 -> img1)
                img2_to_img1 = []

                for idx, (dist, ind) in enumerate(zip(distances_reverse, indices_reverse)):
                    if len(dist) > 1 and dist[0] < ratio_threshold * dist[1] and dist[0] < max_distance:
                        img2_to_img1.append((idx, ind[0], dist[0]))

                # Cross-check: keep only matches that are mutual
                cross_checked = []
                for i_idx, j_idx, dist in img1_to_img2:
                    # Check if the match is reciprocal
                    if any(j_match[0] == j_idx and j_match[1] == i_idx for j_match in img2_to_img1):
                        cross_checked.append((i_idx, j_idx, dist))

                filtered_matches = cross_checked
            else:
                filtered_matches = img1_to_img2

            # Convert indices to keypoints
            pair_key = (img1, img2)
            matches[pair_key] = []

            for i_idx, j_idx, _ in filtered_matches:
                src_kp = tuple(kp1[i_idx])
                dst_kp = tuple(kp2[j_idx])
                matches[pair_key].append((src_kp, dst_kp))

    return matches
