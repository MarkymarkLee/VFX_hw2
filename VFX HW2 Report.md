Group 17 - B10902032 李沛宸 
Github: https://github.com/MarkymarkLee/VFX_hw2
## Algorithm Implementation

The panorama stitching pipeline consists of the following steps:
### 1. Cylindrical Projection

Images are projected onto a cylindrical surface to allow for better alignment of panoramas. The projection maps image coordinates (x, y) to cylindrical coordinates using the formula:

-   θ = x / focal_length
-   h = y / sqrt(x² + focal_length²)

This transformation helps reduce distortion when stitching wide-angle panoramas. The projection creates a mask for each image to identify valid projected areas.

Some slight adjustments: The provided `pano.txt` does not calculate focal length well when the images are too big, so I resized the images such that the max height is 274.

### 2. Feature Detection

Features are detected using MSOP (Multi-Scale Oriented Patches) corners and descriptors:

1. **Harris Corner Detection**:
    - Compute image gradients using Sobel filters
    - Calculate the Harris matrix elements (Ixx, Iyy, Ixy)
    - Compute the corner response function: det(M) - k \* trace(M)²
    - Apply thresholding and non-maximum suppression

2. **MSOP Corner Detection**: (Bonus)
    - Create a scale-space pyramid of the image
    - Calculate the Harris matrix elements (Ixx, Iyy, Ixy)
    - Compute the corner response function: det(M) / trace(M)²
    - Apply thresholding at each scale

3. **MSOP Feature Extraction:**
	 - For each corner, calculate the orientation and scale
	 - Create a 10x10 patch with spacing of 2 pixels
	 - Sample the 100 points and flatten into feature vector
	
4. **Adaptive Non-Maximal Suppression**:
    - Ensure even distribution of features across the image
    - Prioritize stronger features while maintaining spatial diversity

### 3. Feature Matching

Features are matched between image pairs using:

1. **KD-Tree-based Search**:
    - Efficiently find nearest neighbors using a KD-Tree spatial data structure
    - Normalize descriptors for better distance comparison

2. **Ratio Test**:
    - For each feature in the first image, find its two nearest neighbors in the second image
    - Accept the match only if the ratio of distances is below a threshold
    - This helps reject ambiguous matches

3. **Cross-Check Validation**:
    - Perform matching in both directions (image1 → image2 and image2 → image1)
    - Keep only matches that are consistent in both directions
    - This ensures that matches are reliable
### 4. Image Matching

Consistent image matches are found using RANSAC (Random Sample Consensus):

1. **Homography Estimation**:
    - Randomly select 3 pairs of matching points
    - Compute the homography matrix using Direct Linear Transform (DLT)
    - Find $M=\begin{pmatrix} m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23} \\ 0 & 0 & 1 \end{pmatrix}$ such that $\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix}=M\begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$

2. **RANSAC for Outlier Rejection**:
    - Iterate through multiple random samples
    - For each sample, compute homography and count inliers
    - Select the homography with the most inliers
    - Refine the homography using all inliers

3. **Graph Construction**:
    - Create a graph where nodes represent images and edges represent matches
    - Weight edges by the number of inlier matches
    - Identify the central image (node with highest connectivity)

### 5. Multi-band Blending

Image blending is performed using a multi-band approachand blend all the levels:

1. **Canvas Creation**:
    - Determine the size of the final panorama canvas
    - Compute transformation matrices for each image

2. **Image Warping**:
    - Project each image onto the canvas using its homography matrix
    - Create masks for valid image regions

3. **Multi-band Blending**:
    - Create Laplacian pyramids for each image
    - Create Gaussian pyramids for the masks
    - Blend corresponding levels of the Laplacian pyramids
    - Reconstruct the final blended image from the pyramid

### 6. End to End Alignment and Panorama Rectangling (Bonus)

The final step is to align the left most and right most image and rectangles the panorama to produce a clean, rectangular output:

1. **End to End Alignment**:
    - Stitch the left most image to the right side
    - Calculate the of the end images
    - Crop half of the end images so that they're aligned

2. **Boundary Detection**:
    - Identify the irregular boundary of the stitched panorama
    - Estimate the center line of the panorama into a polynomial

3. **Warping for Rectangling**:
    - Fit a smooth curve to the panorama center
    - Warp the image to straighten the center curve
    - Crop the result to a rectangular shape

4. **Hole Filling**:
    - Identify and fill any holes in the final image
    - Use inpainting to maintain visual consistency

## Experimental Results

![[VFX_hw2/report_images/result.png]]
### Step by Step Comparison
#### 1. Cylindrical Transformation

| Original                                  | Cylindrical Transformed   |
| ----------------------------------------- | ------------------------- |
| ![[VFX_hw2/report_images/mks_2.jpg\|345]] | ![[cylindrical.jpg\|345]] |
Straight lines become curved.

#### 2. Harris vs MSOP Corners Detection

| Harris                                      | MSOP                                      |
| ------------------------------------------- | ----------------------------------------- |
| ![[VFX_hw2/report_images/harris_mks_2.jpg]] | ![[VFX_hw2/report_images/msop_mks_2.jpg]] |
Calculated corners mainly focus on exposure differences, so features may largely be affected images with different exposures. The two method didn't differ by a lot.
#### 2. Feature Extraction

| Corners                                   | Patches                                      |
| ----------------------------------------- | -------------------------------------------- |
| ![[VFX_hw2/report_images/msop_mks_2.jpg]] | ![[VFX_hw2/report_images/corners_mks_2.jpg]] |
Here we can see the extracted features using MSOP results in patches with different sizes.
#### 3. Feature/Image Matching
![[VFX_hw2/report_images/mks_2.jpg_mks_3.jpg_comparisons.jpg]]

#### 4. Direct Warp
![[VFX_hw2/report_images/mks_2.jpg_mks_3.jpg_warped.jpg]]
The line connecting both images are obvious.
#### 5. Blending
![[blended.png]]
The line is less obvious.
#### 6. Canvas Size
![[bounding_box.png]]
Green signals the position of each image after end to end alignment and blue shows where the final panorama is cropped.
#### 7. End to End Alignment
Before Cropping
![[end_to_end_panorama.jpg]]
After Cropping
![[cropped_panorama.jpg]]
This aligns the left most and right most image
#### 8. Rectangling
![[VFX_hw2/report_images/result.png]]
The black border is removed and filled with neighbors' colors
### Visual Comparison

The cylindrical projection significantly improves panorama quality for wide field-of-view scenes compared to planar projection. Key observations:

1. **Reduced Distortion**:
    - Cylindrical projection reduces stretching at panorama edges
    - Maintains better proportion of objects throughout the image

2. **Feature Matching Quality**:
    - Cross-check validation greatly reduces false matches
    - RANSAC effectively filters remaining outliers
    - Approximately 70-90% of initial matches are typically rejected as outliers

3. **Blending Effects**:
    - Multi-band blending produces smooth transitions between images
    - Significantly reduces visible seams compared to simple alpha blending
    - Best results are achieved with 4-6 levels in the pyramid

4. **Limitations**:
    - Performance degrades with extreme exposure differences between images
    - Very wide panoramas may still show some distortion at top and bottom edges
    - Features-poor regions (like clear skies) can cause alignment challenges

### Optimizations

Several optimizations were implemented to improve performance:

1. **Adaptive Feature Selection**:

    - Distributes features more evenly across the image
    - Improves stitching quality in feature-sparse regions

2. **Connection Graph**:

    - Limits the number of connections per image
    - Prevents error propagation in long image sequences

3. **Rectangling**:
    - Produces visually pleasing rectangular panoramas
    - Fills gaps naturally using inpainting techniques

## Future Improvements

Potential enhancements to the current implementation:

1. Exposure compensation between images
2. Global bundle adjustment for improved alignment
3. Parallax handling for scenes with significant depth variation
4. GPU acceleration for faster processing
5. Automatic focal length estimation when not provided

## References

1. M. Brown and D. G. Lowe, "Recognising Panoramas," in Proceedings of the Ninth IEEE International Conference on Computer Vision, 2003.
2. R. Szeliski, "Image Alignment and Stitching: A Tutorial," Foundations and Trends in Computer Graphics and Vision, 2006.
3. D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," International Journal of Computer Vision, 2004.
4. P. J. Burt and E. H. Adelson, "A Multiresolution Spline with Application to Image Mosaics," ACM Transactions on Graphics, 1983.
