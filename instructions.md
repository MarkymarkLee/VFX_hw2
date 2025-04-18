In this project, you will implement parts of the "Recognising Panoramas" paper to create a panorama image from several images.

There are basically five components in that paper.

1. feature detection
2. feature matching
3. image matching
4. bundle adjustment
5. blending (Gradient-domain stitching、poisson image blending)
6. End-to-end alignment
7. Rectangling

For feature detection and matching, we want to use harris Corner Detection and MSOP (Multi-Scale Oriented Patches).

Note that you need to implement everything from scratch. It is not allowed to use exisiting feature libraries.

You should create a python project, and a virtual environment using conda with all the dependencies written in `requirements.txt`. Note that you don't need to specify the versions.

`main.py` in the root of this directory is the entry point and takes in some important parameters including the input folder.
Inside `main.py` should be a function `create_panoramas` that ensembles all the functions.

All the utility functions should be put under `utils/` and you should separate each component into their own files.

Data can be sampled from `data/parrinton` the file `pano.txt`

```
The file looks likethe following:
C:\Users\cyy\Desktop\autostitch\images\test\100-0024_img.jpg
568 758

1 0 378.5
0 1 283.5
0 0 1

0.989768 -0.0039487 0.142633
0.00390831 0.999992 0.000563158
-0.142635 1.74769e-008 0.989775

897.93

...
where 897.93 is the estimated focal length for the first image.
```

You should use this information to create cylindrical view panoramas.
