import os
import numpy as np
import cv2
from PIL import Image

from utils.cylindrical_projection import project_to_canvas


def GaussianPyramid(img, level):
    GP = [img]
    for i in range(level - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def create_gaussian_pyramid(img, mask, levels=6):
    """
    Create a Gaussian pyramid for an image

    Args:
        img: Input image
        mask: Binary mask for the image
        levels: Number of pyramid levels

    Returns:
        pyramid: List of images in Gaussian pyramid
    """
    mask = mask > 0
    mask = mask.astype(np.float32)
    img[mask == 0] = 0
    pyramid = [img.astype(np.float32)]
    mask_pyramid = [(mask > 0)]
    for i in range(levels - 1):
        img = cv2.pyrDown(img).astype(np.float32)
        mask = cv2.pyrDown(mask).astype(np.float32)
        try:
            img = img / (mask[:, :, np.newaxis] + 1e-8)
        except:
            img = img / (mask[:, :] + 1e-8)
        img[mask == 0] = 0
        mask = (mask > 0).astype(np.float32)
        mask_pyramid.append((mask > 0))
        pyramid.append(img.astype(np.float32))
    return pyramid, mask_pyramid


def create_laplacian_pyramid(gaussian_pyramid, mask_pyramid):
    """
    Create a Laplacian pyramid from a Gaussian pyramid

    Args:
        gaussian_pyramid: Gaussian pyramid
        mask_pyramid: Mask pyramid

    Returns:
        pyramid: Laplacian pyramid
    """
    levels = len(gaussian_pyramid)
    laplacian_pyramid = []

    for i in range(levels - 1):
        curr_h, curr_w = gaussian_pyramid[i].shape[:2]
        upsampled = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(curr_w, curr_h))
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        mask = mask_pyramid[i] > 0
        laplacian[mask == 0] = 0
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def blend_two_images(img1, mask1, img2, mask2, levels=6):
    """
    Blend two images using multiband blending

    Args:
        img1: First image
        mask1: Binary mask for the first image
        img2: Second image
        mask2: Binary mask for the second image
        levels: Number of pyramid levels

    Returns:
        blended: Blended image
    """
    # Ensure images and mask have the same shape
    assert img1.shape == img2.shape, "Images must have the same shape"
    assert img1.shape[:2] == mask1.shape[:2], "Mask must have the same dimensions as images"
    assert img2.shape[:2] == mask2.shape[:2], "Mask must have the same dimensions as images"

    img1_gaussian_pyramid, mask1_gaussian_pyramid = create_gaussian_pyramid(
        img1, mask1, levels)
    img2_gaussian_pyramid, mask2_gaussian_pyramid2 = create_gaussian_pyramid(
        img2, mask2, levels)

    # Create Laplacian pyramids for the images
    img1_laplacian = create_laplacian_pyramid(
        img1_gaussian_pyramid, mask1_gaussian_pyramid)
    img2_laplacian = create_laplacian_pyramid(
        img2_gaussian_pyramid, mask2_gaussian_pyramid2)

    # Create Gaussian pyramid for the masks
    blend_mask_gaussian_pyramid = create_gaussian_pyramid(
        (mask1 > 0).astype(np.float32), mask1 | mask2, levels)[0]

    # Blend the Laplacian pyramids using the Gaussian mask pyramid
    blended_pyramid = []
    for i in range(levels):
        blend_mask = blend_mask_gaussian_pyramid[i]
        blend_mask = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
        blended = img1_laplacian[i] * blend_mask + \
            img2_laplacian[i] * (1 - blend_mask)
        blended_pyramid.append(blended)

    # Reconstruct the blended image from the blended pyramid
    blended = blended_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        blended = cv2.pyrUp(blended, dstsize=(
            blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        blended = cv2.add(blended, blended_pyramid[i])

    # Ensure the blended image is in valid range
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    final_mask = mask1 | mask2
    blended[final_mask == 0] = 0

    return blended


def create_image_graph(consistent_matches):
    """
    Create a graph representing the connections between images without using NetworkX

    Args:
        consistent_matches: Dictionary with image pairs as keys and
                          (homography, inlier_matches) as values

    Returns:
        graph: Dictionary with images as keys and lists of connected images as values
        central_node: Node with highest degree (most connections)
    """
    # Create an adjacency list representation of the graph
    graph = {}

    # Add edges from consistent matches
    for (img1, img2), (H, inliers) in consistent_matches.items():
        if img1 not in graph:
            graph[img1] = []
        if img2 not in graph:
            graph[img2] = []

        # Add bidirectional connections
        graph[img1].append((img2, len(inliers), np.linalg.inv(H)))
        # Store the inverse homography for the reverse direction
        graph[img2].append((img1, len(inliers), H))

    # Find the node with the highest degree (most connections)
    if graph:
        central_node = max(graph, key=lambda x: len(graph[x]))
    else:
        central_node = None

    return graph, central_node


def create_blending_order(graph, central_node):
    """
    Create a blending order starting from the most connected image

    Args:
        graph: Dictionary with images as keys and lists of connected images as values
        central_node: Node with highest degree (most connections)

    Returns:
        order: List of images in blending order
    """
    if central_node is None:
        return []

    # Start with the central node
    order = [(central_node, None, None)]
    visited = {central_node}

    # Use BFS to build the order, prioritizing by number of connections
    while len(visited) < len(graph):
        next_nodes = []
        for node in visited:
            neighbors = [neighbor for neighbor, _, _ in graph[node]]
            for i, neighbor in enumerate(neighbors):
                if neighbor not in visited and neighbor not in next_nodes:
                    next_nodes.append((neighbor, node, graph[node][i][2]))

        # Sort neighbors by their degree (number of connections)
        def count_connections(next):
            return sum([inliers for _, inliers, _ in graph.get(next[0], [])])
        next_nodes.sort(key=count_connections, reverse=True)

        # Add the highest degree neighbor to the order and mark as visited
        if next_nodes:
            order.append(next_nodes[0])
            visited.add(next_nodes[0][0])
        else:
            # If no neighbors found, break to avoid infinite loop
            break

    # Add end to end_alignment
    last_img = order[-1][0]
    last_father = order[-1][1]
    for connections in graph[last_img]:
        if connections[0] != last_father:
            order.append((f"{connections[0]}_L", last_img, connections[2]))
            print(f"Adding {connections[0]} to the end of the order")
            break

    return order


def calc_canvas(homographies, blend_order, image_dict):
    all_pts = np.zeros((len(blend_order), 4, 2), dtype=np.float32)
    for j, (image_name, _, _) in enumerate(blend_order):
        if image_name.endswith("_L"):
            cur_img, cur_mask = image_dict[image_name[:-2]]
        else:
            cur_img, cur_mask = image_dict[image_name]
        h, w = cur_img.shape[:2]
        pts = np.array(
            [[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]])
        cur_homography = homographies[image_name]
        warped_pts = (cur_homography @ pts.T).T
        warped_pts = warped_pts / np.abs(warped_pts[:, 2:])
        warped_pts = warped_pts[:, :2]
        all_pts[j] = warped_pts[:, :2]
    return all_pts


def draw_bounding_box(best_size, best_homographies, translation, image_dict, blend_order, cropped_box, output_folder):
    # draw the bounding box
    canvas = np.zeros((best_size[1], best_size[0], 3), dtype=np.uint8)
    for i, (image_name, _, _) in enumerate(blend_order):
        if image_name.endswith("_L"):
            cur_img, cur_mask = image_dict[image_name[:-2]]
        else:
            cur_img, cur_mask = image_dict[image_name]
        h, w = cur_img.shape[:2]
        pts = np.array(
            [[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]])
        cur_homography = best_homographies[image_name]
        warped_pts = (cur_homography @ pts.T).T
        warped_pts = warped_pts / np.abs(warped_pts[:, 2:])
        warped_pts = warped_pts[:, :2]
        warped_pts[:, 0] += translation[0]
        warped_pts[:, 1] += translation[1]
        cv2.polylines(canvas, [np.int32(warped_pts)], True, (0, 255, 0), 1)
        cv2.putText(canvas, image_name, (int(warped_pts[0][0]), int(warped_pts[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # draw the cropped box
    cv2.rectangle(canvas, (cropped_box[0], cropped_box[1]),
                  (cropped_box[0] + cropped_box[2], cropped_box[1] + cropped_box[3]), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(output_folder, "6_bounding_box.png"), canvas)


def get_canvas_data(image_dict, blend_order, output_folder, draw):
    """
    Get the canvas size for blending images

    Args:
        image_dict: Dictionary with image names and their data
        blend_order: List of images in blending order

    Returns:
        (height, width): Size of the canvas
        translation_matrix: Translation matrix for the canvas
    """
    homographies = {}
    # Get the bounding box of the current panorama and next image
    for i, (cur_image_name, father, homography) in enumerate(blend_order):
        if father is None:
            cur_homography = np.eye(3)
        else:
            cur_homography = homography @ homographies[father]
        homographies[cur_image_name] = cur_homography

    all_pts = calc_canvas(homographies, blend_order, image_dict)
    min_x = int(np.min(all_pts[:, :, 0]))
    min_y = int(np.min(all_pts[:, :, 1]))
    max_x = int(np.max(all_pts[:, :, 0]))
    max_y = int(np.max(all_pts[:, :, 1]))

    width = max_x - min_x
    height = max_y - min_y

    size = (width, height)
    translation = np.array([-min_x, -min_y], dtype=np.float32)

    centers = np.mean(all_pts[:, :, :2], axis=1)

    if width > height:
        # calc cropped
        cropped_min_x = int(np.min(centers[:, 0]))
        cropped_max_x = int(np.max(centers[:, 0]))

        cropped_width = cropped_max_x - cropped_min_x
        cropped_height = height

        cropped_box = (cropped_min_x-min_x, 0, cropped_width, cropped_height)

    else:
        # calc cropped
        cropped_min_y = int(np.min(centers[:, 1]))
        cropped_max_y = int(np.max(centers[:, 1]))

        cropped_width = width
        cropped_height = cropped_max_y - cropped_min_y

        cropped_box = (0, cropped_min_y-min_y, cropped_width, cropped_height)
    # draw the bounding box
    if draw:
        draw_bounding_box(size, homographies, translation,
                          image_dict, blend_order, cropped_box, output_folder)

    return {
        "canvas_size": size,
        "translation": translation,
        "homographies": homographies,
        "cropped_box": cropped_box,
    }


def blend_images(images, consistent_matches, output_folder, draw):
    """
    Blend images to create a panorama using multiband blending

    Args:
        images: List of (image_name, image_data)
        consistent_matches: Dictionary with image pairs as keys and
                          (homography, inlier_matches) as values

    Returns:
        panorama: Blended panorama image
    """
    if not images or not consistent_matches:
        # If no matches or images, return the first image
        if images:
            return Image.fromarray(images[0][1])
        return None

    # Create image name to image data mapping
    image_dict = {name: (img, mask) for name, img, mask in images}

    # Create a graph to find the blending order
    graph, central_node = create_image_graph(consistent_matches)

    # If no central node found (e.g., no matches), return the first image
    if central_node is None:
        return Image.fromarray(images[0][1])

    # Create blending order starting from the most connected image
    blend_order = create_blending_order(graph, central_node)
    print(f"Blending order: {[(img[0], img[1]) for img in blend_order]}")

    data = get_canvas_data(image_dict, blend_order, output_folder, draw)
    width, height = data["canvas_size"]
    translation = data["translation"]
    homographies = data["homographies"]
    cropped_box = data["cropped_box"]

    print(f"Canvas size: {width}x{height}")

    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    panorama_mask = np.zeros((height, width), dtype=bool)

    for image_name, _, _ in blend_order:
        print(f"Blending {image_name}...", end='\r')
        homography = homographies[image_name]

        if image_name.endswith("_L"):
            image, mask = image_dict[image_name[:-2]]
        else:
            image, mask = image_dict[image_name]

        warped_image, warped_mask = project_to_canvas(
            image, mask,
            homography=homography, translation=translation, canvas_size=(width, height))

        # Blend the images using multiband blending
        panorama = blend_two_images(
            panorama, panorama_mask, warped_image, warped_mask)

        panorama_mask = panorama_mask | warped_mask
        if draw:
            if image_name.endswith("_L"):
                cv2.imwrite(os.path.join(output_folder,
                            "7_end_to_end_panorama.jpg"), panorama)
            else:
                cv2.imwrite(os.path.join(output_folder, "5_blended_images",
                            f"blended_{image_name}"), panorama)

    cropped_panorama = np.zeros(
        (cropped_box[3], cropped_box[2], 3), dtype=np.uint8)
    cropped_panorama = panorama[cropped_box[1]:cropped_box[1] + cropped_box[3],
                                cropped_box[0]:cropped_box[0] + cropped_box[2]]

    cropped_panorama_mask = np.zeros(
        (cropped_box[3], cropped_box[2]), dtype=bool)
    cropped_panorama_mask = panorama_mask[cropped_box[1]:cropped_box[1] + cropped_box[3],
                                          cropped_box[0]:cropped_box[0] + cropped_box[2]]

    print("Blending complete.            ")
    print(f"Panorama size: {panorama.shape[1]}x{panorama.shape[0]}")
    print(
        f"Cropped panorama size: {cropped_panorama.shape[1]}x{cropped_panorama.shape[0]}")

    return cropped_panorama, cropped_panorama_mask
