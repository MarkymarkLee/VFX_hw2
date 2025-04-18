import argparse
import os
from utils.feature_detection import detect_features
from utils.feature_matching import match_features
from utils.image_matching import match_images
from utils.bundle_adjustment import adjust_bundle
from utils.blending import blend_images
from utils.alignment import align_end_to_end
from utils.rectangling import rectangle_panorama
from utils.cylindrical_projection import cylindrical_project


def create_panoramas(input_folder, output_file, focal_length=None):
    """
    Main function to create panorama from multiple images

    Args:
        input_folder (str): Path to folder containing input images
        output_file (str): Path to save the output panorama
        focal_length (float, optional): Focal length for cylindrical projection.
                                      If None, will try to read from pano.txt
    """
    # Check if pano.txt exists in the input folder
    pano_txt_path = os.path.join(input_folder, "pano.txt")
    focal_lengths = {}

    if os.path.exists(pano_txt_path) and focal_length is None:
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

    # Get all images from the input folder
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images in {input_folder}")

    # Apply cylindrical projection if focal length is available
    warped_images = []
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        f = focal_lengths.get(img_file, focal_length)
        if f is not None:
            warped_img = cylindrical_project(img_path, f)
            warped_images.append((img_file, warped_img))
        else:
            print(
                f"No focal length found for {img_file}, skipping cylindrical projection")

    # 1. Detect features in all images
    print("Detecting features...")
    features = {}
    for img_file, img in warped_images:
        features[img_file] = detect_features(img)

    # 2. Match features between image pairs
    print("Matching features...")
    matches = match_features(features)

    # 3. Find consistent image matches
    print("Finding consistent image matches...")
    consistent_matches = match_images(matches)

    # 4. Bundle adjustment to optimize camera parameters
    print("Optimizing camera parameters...")
    camera_params = adjust_bundle(consistent_matches)

    # 5. Perform image blending
    print("Blending images...")
    panorama = blend_images(warped_images, camera_params)

    # 6. End-to-end alignment
    print("Performing end-to-end alignment...")
    aligned_panorama = align_end_to_end(panorama)

    # 7. Rectangle the panorama
    print("Rectangling panorama...")
    final_panorama = rectangle_panorama(aligned_panorama)

    # Save the result
    final_panorama.save(output_file)
    print(f"Panorama saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create panorama from multiple images')
    parser.add_argument('--input', '-i', required=True,
                        help='Input folder containing images')
    parser.add_argument('--output', '-o', required=True,
                        help='Output panorama file')
    parser.add_argument('--focal_length', '-f', type=float,
                        help='Focal length for cylindrical projection')

    args = parser.parse_args()

    create_panoramas(args.input, args.output, args.focal_length)


if __name__ == "__main__":
    main()
