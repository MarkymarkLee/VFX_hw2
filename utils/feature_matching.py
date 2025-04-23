import numpy as np
from scipy.spatial import KDTree


def match_features(features_dict):
    """
    Match features between image pairs using ratio test

    Args:
        features_dict: Dictionary with image filenames as keys and feature dictionaries as values, file dictionaries have 2 keys: "keypoints" and "descriptors"

    Returns:
        matches: Dictionary with image pairs as keys and matched keypoints as values
    """
    # Initialize matches dictionary
    matches = {}

    # Get list of image names
    image_names = list(features_dict.keys())

    # Combine all descriptors and keep track of their image and keypoint indices
    all_descriptors = []
    # Store (image_name, keypoint_idx) for each descriptor
    descriptor_sources = []

    for img_name in image_names:
        img_features = features_dict[img_name]
        descriptors = img_features['descriptors']

        if not isinstance(descriptors, np.ndarray):
            descriptors = np.array(descriptors)

        for idx, desc in enumerate(descriptors):
            all_descriptors.append(desc)
            descriptor_sources.append((img_name, idx))

    # Convert to numpy array for KD-tree
    all_descriptors = np.array(all_descriptors)

    # Create KD-tree with all descriptors from all images
    tree = KDTree(all_descriptors)

    # For each descriptor, find the 4 closest descriptors across all images
    k = 5  # 5 instead of 4 because the closest will be the descriptor itself
    _, indices = tree.query(all_descriptors, k=k)

    # Process matches
    for i, neighbors in enumerate(indices):
        source_img, source_idx = descriptor_sources[i]
        source_kp = features_dict[source_img]['keypoints'][source_idx]

        # Skip the first match (which is the descriptor itself)
        for j in neighbors:  # Skip the first match (self)
            target_img, target_idx = descriptor_sources[j]

            # Skip if it's the same image
            if source_img == target_img:
                continue

            target_kp = features_dict[target_img]['keypoints'][target_idx]

            # Ensure consistent order of image pairs in the key
            if source_img < target_img:
                pair_key = (source_img, target_img)
                kp_pair = (tuple(source_kp), tuple(target_kp))
            else:
                pair_key = (target_img, source_img)
                kp_pair = (tuple(target_kp), tuple(source_kp))

            # Initialize list for this pair if it doesn't exist
            if pair_key not in matches:
                matches[pair_key] = []

            matches[pair_key].append(kp_pair)

    for pair_key in matches:
        matches[pair_key] = list(set(matches[pair_key]))

    return matches
