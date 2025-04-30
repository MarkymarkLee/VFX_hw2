import argparse
import os
import cv2
import numpy as np
import shutil
from utils.blending import blend_images
from utils.cylindrical_projection import cylindrical_project
from utils.draw_comparisons import draw_comparisons
from utils.feature_detection import detect_features
from utils.feature_matching import match_features
from utils.image_matching import match_images
from utils.rectangling import rectangle_panorama
random = np.random.RandomState(42)


def find_focal_lengths(input_folder):
    pano_txt_path = os.path.join(input_folder, "pano.txt")
    focal_lengths = {}
    assert os.path.exists(pano_txt_path), f"No pano.txt. in {input_folder}"

    # Read focal lengths from pano.txt
    with open(pano_txt_path, 'r') as f:
        lines = f.readlines()

    # Parse focal lengths from pano.txt
    current_image = None
    for line in lines:
        line = line.strip()
        if line.endswith(".jpg") or line.endswith(".JPG"):
            current_image = os.path.basename(line)
        elif current_image is not None and len(line.split()) == 1:
            try:
                if float(line) <= 0:
                    print(
                        f"Skipping invalid focal length {line} for {current_image}.")
                    current_image = None
                    continue
                focal_lengths[current_image] = float(line)
                current_image = None
            except ValueError:
                pass

    print(f"Focal lengths: {focal_lengths}")
    return focal_lengths


def find_images(input_folder):
    """
    Find all images in the input folder

    Args:
        input_folder (str): Path to folder containing input images

    Returns:
        list: List of image file paths
    """
    image_files = []
    for f in os.listdir(input_folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) and "pano" not in f.lower():
            image_files.append(f)
    return image_files


def setup_outputs(output_folder):
    """
    Create output folders for results

    Args:
        output_folder (str): Path to save the output panorama
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "1_cylindrical"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "2_features"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "3_comparisons"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "4_direct_warps"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "5_blended_images"), exist_ok=True)


def create_panoramas(input_folder, output_folder, result_only):
    """
    Main function to create panorama from multiple images

    Args:
        input_folder (str): Path to folder containing input images
        output_file (str): Path to save the output panorama
        focal_length (float, optional): Focal length for cylindrical projection.
                                      If None, will try to read from pano.txt
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    setup_outputs(output_folder)

    focal_lengths = find_focal_lengths(input_folder)

    # Get all images from the input folder
    image_files = find_images(input_folder)
    assert len(image_files) > 0, "No images found in the input folder."
    print(f"Found {len(image_files)} images in {input_folder}")

    # Read images
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        scale = max(img.shape[0], img.shape[1]) / 1024
        if scale > 1:
            img = cv2.resize(img, (0, 0), fx=1/scale, fy=1/scale)
        images.append((img_file, np.array(img)))

    # 0. Cylindrical projection
    # Use a random focal length as the radius
    radius = None
    if focal_lengths and radius is None:
        # Use the first focal length as radius
        radius = list(focal_lengths.values())[0]
    print(f"Performing cylindrical projection with radius {radius}...")
    for i in range(len(images)):
        img_file, img = images[i]
        focal_length = focal_lengths.get(img_file, None)
        if focal_length is None:
            print(
                f"Focal length not found for {img_file}. Using default radius.")
            focal_length = radius
        img, mask = cylindrical_project(img, focal_length)
        images[i] = (img_file, img, mask)
        if not result_only:
            cv2.imwrite(os.path.join(output_folder,
                                     "1_cylindrical", "projected_" + img_file), img)
        # cv2.imwrite(os.path.join(output_folder,
        #             "projected_mask_" + img_file), mask * 255)

    # 1. Detect features in all images
    print("Detecting features...")
    features = {}
    for i, (img_file, img, mask) in enumerate(images):
        print(f"Processing image {i + 1}/{len(images)}", end='\r')
        features[img_file] = detect_features(
            img_file, img, mask, output_folder, draw=not result_only)

    print("\nFeature detection complete.")

    # 2. Match features between image pairs
    print("Matching features...")
    matches = match_features(features)

    # 3. Find consistent image matches
    print("Finding consistent image matches...")
    consistent_matches = match_images(matches)

    if not result_only:
        draw_comparisons(images, consistent_matches, output_folder)

    # 5. Perform image blending
    print("Blending images...")
    panorama, panorama_mask = blend_images(
        images, consistent_matches, output_folder, draw=not result_only)
    cv2.imwrite(os.path.join(output_folder,
                "8_cropped_panorama.jpg"), panorama)

    # 6. Rectangle the panorama
    print("Rectangling the panorama...")
    panorama = rectangle_panorama(panorama, panorama_mask)
    cv2.imwrite(os.path.join(output_folder, "panorama.jpg"), panorama)
    cv2.imwrite("result.png", panorama)


def main():
    parser = argparse.ArgumentParser(
        description='Create panorama from multiple images')
    parser.add_argument('--input', '-i', required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', default='outputs/',
                        help='Output folder')
    parser.add_argument('--result-only', '-r', action='store_true',
                        help='Only show the result image')
    args = parser.parse_args()

    create_panoramas(args.input, args.output, args.result_only)


if __name__ == "__main__":
    main()
