import argparse
import os
from PIL import Image
import numpy as np
import shutil
from utils.cylindrical_projection import cylindrical_project
from utils.feature_detection import detect_features
from utils.feature_matching import match_features
from utils.image_matching import match_images
from utils.bundle_adjustment import adjust_bundle
from utils.blending import blend_images
from utils.alignment import align_end_to_end
from utils.rectangling import rectangle_panorama


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


def create_panoramas(input_folder, output_folder):
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
    os.makedirs(output_folder, exist_ok=True)

    focal_lengths = find_focal_lengths(input_folder)

    # Get all images from the input folder
    image_files = find_images(input_folder)
    assert len(image_files) > 0, "No images found in the input folder."
    print(f"Found {len(image_files)} images in {input_folder}")

    # Read images and convert to RGB
    images = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path).convert("RGB")
        images.append((img_file, np.array(img)))

    # 0. Cylindrical projection
    # Use a random focal length as the radius
    radius = None
    if focal_lengths and radius is None:
        # Use the first focal length as radius
        radius = list(focal_lengths.values())[0] + 1.0
    print(f"Performing cylindrical projection with radius {radius}...")
    for i in range(len(images)):
        img_file, img = images[i]
        focal_length = focal_lengths.get(img_file, None)
        assert focal_length is not None, f"Focal length not found for {img_file}"
        img = cylindrical_project(img, focal_length, radius)
        images[i] = (img_file, img)

    # 1. Detect features in all images
    print("Detecting features...")
    features = {}
    for img_file, img in images:
        print(f"Processing {img_file}...", end='\r')
        features[img_file] = detect_features(
            img_file, img, output_folder, draw=True)

    print("\nFeature detection complete.")

    # 2. Match features between image pairs
    print("Matching features...")
    matches = match_features(features, )

    # 3. Find consistent image matches
    print("Finding consistent image matches...")
    consistent_matches = match_images(matches)

    # 4. Bundle adjustment to optimize camera parameters
    print("Optimizing camera parameters...")
    camera_params = adjust_bundle(consistent_matches)

    # 5. Perform image blending
    print("Blending images...")
    panorama = blend_images(images, camera_params)

    # 6. End-to-end alignment
    print("Performing end-to-end alignment...")
    aligned_panorama = align_end_to_end(panorama)

    # 7. Rectangle the panorama
    print("Rectangling panorama...")
    final_panorama = rectangle_panorama(aligned_panorama)

    # Save the result
    final_panorama.save(f"{output_folder}/result.jpg")
    print(f"Panorama saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Create panorama from multiple images')
    parser.add_argument('--input', '-i', required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', default='outputs/',
                        help='Output folder')
    args = parser.parse_args()

    create_panoramas(args.input, args.output)


if __name__ == "__main__":
    main()
