import cv2
import numpy as np


def align_end_to_end(panorama):
    """
    Apply end-to-end alignment to eliminate the drift that can occur
    in panorama stitching

    Args:
        panorama: Stitched panorama image

    Returns:
        aligned_panorama: Panorama with end-to-end alignment
    """
    # For end-to-end alignment, we need to detect and match features
    # between the left and right edges of the panorama

    # Get dimensions
    height, width = panorama.shape[:2]

    # If the panorama doesn't wrap around (e.g., not 360 degrees), return as is
    # A simple heuristic: check if both edges have non-zero content
    left_edge = panorama[:, :50, :]
    right_edge = panorama[:, -50:, :]

    if np.mean(left_edge) < 10 or np.mean(right_edge) < 10:
        # One of the edges is mostly black, so not a full wrap-around
        return panorama

    # Extract left and right strips for feature matching
    left_strip = panorama[:, :100, :]
    right_strip = panorama[:, -100:, :]

    # Convert to grayscale for feature detection
    left_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY)

    # Detect features on both edges
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(left_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(right_gray, None)

    # Match features
    if descriptors1 is not None and descriptors2 is not None and len(keypoints1) > 0 and len(keypoints2) > 0:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract matched points
        if len(good_matches) >= 4:
            src_pts = np.float32(
                [keypoints1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32(
                [keypoints2[m.trainIdx].pt for m in good_matches])

            # Adjust x-coordinates for right strip
            dst_pts[:, 0] += width - 100

            # Find the vertical transformation using RANSAC
            _, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if inliers is not None and np.sum(inliers) >= 4:
                # Extract vertical translation
                src_y = np.mean([src_pts[i][1]
                                for i, val in enumerate(inliers) if val])
                dst_y = np.mean([dst_pts[i][1]
                                for i, val in enumerate(inliers) if val])

                # Calculate vertical offset
                v_offset = dst_y - src_y

                # Create a new canvas with the adjusted height
                new_height = height + abs(int(v_offset))
                aligned = np.zeros((new_height, width, 3), dtype=np.uint8)

                # Copy the panorama to the adjusted position
                if v_offset > 0:
                    aligned[int(v_offset):, :, :] = panorama
                else:
                    aligned[:height, :, :] = panorama

                return aligned

    # If alignment fails, return the original panorama
    return panorama
