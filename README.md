# Cylindrical Panorama Stitching

This project implements cylindrical panorama stitching from scratch, based on the "Recognising Panoramas" paper. The implementation creates panorama images from multiple input images by detecting features, matching them across images, and stitching them together.

## Usage

### Requirements

-   Python 3.7+
-   Required packages (install via `pip install -r requirements.txt`):
    -   numpy
    -   opencv-python
    -   matplotlib
    -   scipy
    -   pillow

### Running the Code

To generate my result, run `python main.py -i data/mks -r`

To create a panorama, use the following command:

```bash
python main.py --input <input_folder> --output <output_folder> -r
```

-   `<input_folder>`: Directory containing source images and a `pano.txt` file with focal length information
-   `<output_folder>`: Directory where output files will be saved (default: `outputs/`)
-   `-r` (optional): Adding this tag will only create a result.png under the current folder.

Example:

```bash
python main.py --input data/parrington --output outputs/
```

### Input Format

The input folder should contain:

1. A set of images (JPG, PNG, etc.)
2. A `pano.txt` file with focal length information for each image (the `pano.txt` file can be created using the attached `autostitch.exe`)
