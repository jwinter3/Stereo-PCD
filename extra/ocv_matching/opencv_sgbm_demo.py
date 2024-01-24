import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# imgL = cv.imread("data/middlebury2001/scene1.row3.col1.png", cv.IMREAD_GRAYSCALE)
# imgR = cv.imread("data/middlebury2001/scene1.row3.col3.png", cv.IMREAD_GRAYSCALE)

# imgL = cv.imread("data/middlebury2021/im0.png", cv.IMREAD_GRAYSCALE)
# imgR = cv.imread("data/middlebury2021/im1.png", cv.IMREAD_GRAYSCALE)

imgL = cv.imread("000000_10.png", cv.IMREAD_UNCHANGED)
imgR = cv.imread("000000_10.png", cv.IMREAD_UNCHANGED)


# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 1
num_disp = 112 - min_disp
stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=16,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Normalize the values to a range from 0..255 for a grayscale image
disparity = cv.normalize(
    disparity, disparity, alpha=255, beta=0, norm_type=cv.NORM_MINMAX
)
disparity = np.uint8(disparity)

plt.imshow(disparity, "gray")
plt.savefig("disparity_SGBM_norm.png")
