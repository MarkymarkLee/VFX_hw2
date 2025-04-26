import cv2
import numpy as np


def draw_comparisons(images, consistent_matches):
    """
    Draw comparisons between images based on consistent matches.

    Args:
        images: List of tuples (image_file, image_array).
        consistent_matches: Dictionary of consistent matches.
    """
    image_dict = {img_file: img for img_file, img, _ in images}
    for pairs, (H, keypoints) in consistent_matches.items():
        img1 = image_dict[pairs[0]]
        img2 = image_dict[pairs[1]]

        # Ensure both images have the same height by padding or resizing
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2:
            if h1 > h2:
                padding = h1 - h2
                img2 = cv2.copyMakeBorder(
                    img2, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            else:
                padding = h2 - h1
                img1 = cv2.copyMakeBorder(
                    img1, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        canvas = np.hstack((img2, img1))
        img_file = f"{pairs[0]}_{pairs[1]}"

        print(f"Drawing comparisons for: {img_file}")
        for pt1, pt2 in keypoints:
            left_point = (int(pt1[0]), int(pt1[1]))
            right_point = (int(pt2[0] + img1.shape[1]), int(pt2[1]))
            # print(f"Left point: {left_point}, Right point: {right_point}")
            # Draw lines between the points
            cv2.line(canvas, left_point, right_point,
                     (0, 255, 0), 1, cv2.LINE_AA)
            # Draw circles at the points
            cv2.circle(canvas, left_point, 5, (0, 0, 255), -1)
            cv2.circle(canvas, right_point, 5, (255, 0, 0), -1)

        # Save or display the image with drawn comparisons
        cv2.imwrite(f"outputs/{img_file}_comparisons.jpg", canvas)

        # warp the images using the homography matrix

        # 1. calculate the size of the output image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.array([[0, 0, 1], [w1, 0, 1], [w1, h1, 1], [0, h1, 1]])
        pts2 = np.array([[0, 0, 1], [w2, 0, 1], [w2, h2, 1], [0, h2, 1]])
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        # Convert H to float32 if it's not already
        H_float = np.float32(H)

        pts1 = pts1 @ H_float.T
        pts1 = pts1 / pts1[:, 2:]
        pts1 = pts1[:, :2]
        pts2 = pts2[:, :2]
        all_pts = np.concatenate((pts1, pts2), axis=0)
        min_x = int(np.min(all_pts[:, 0]))
        min_y = int(np.min(all_pts[:, 1]))
        max_x = int(np.max(all_pts[:, 0]))
        max_y = int(np.max(all_pts[:, 1]))
        width = max_x - min_x
        height = max_y - min_y
        print(f"Output image size: {width}x{height}")
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas.fill(255)

        # 2. calculate the translation matrix
        translation_matrix = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

        # 3. warp the images using the homography matrix and translation matrix
        canvas = cv2.warpPerspective(
            img1, translation_matrix @ H_float, (width, height), dst=canvas)
        canvas = cv2.warpPerspective(
            img2, translation_matrix @ np.eye(3), (width, height), dst=canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        # Save or display the image with drawn comparisons
        cv2.imwrite(f"outputs/{img_file}_warped.jpg", canvas)
