import cv2 as cv
import numpy as np
from stereo_pcd.evaluation.eval_image import (
    calc_stats_pixels,
    stat_pixels_to_percentages,
)
from stereo_pcd.datasets.kitti_stereo_dataset import KittiStereoDataset

dataset = KittiStereoDataset("...")
sample = dataset.read_sample("000006")

gt, imgL, imgR = sample.gt_disp, sample.left_img, sample.right_img

window_size = 3

stereo = cv.StereoSGBM_create(
    preFilterCap=63,
    P1=4 * window_size**2,
    P2=32 * window_size**2,
    minDisparity=1,
    numDisparities=128,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    mode=cv.STEREO_SGBM_MODE_HH,
)

read_disp = stereo.compute(imgL, imgR)
disparity = read_disp.astype(np.float32) / 16

mae, rmse, good_pixel_rate, *_ = stat_pixels_to_percentages(
    *calc_stats_pixels(
        disparity,
        gt,
        invalid_value=0,
    ),
    eval_only_valid_pixel=True,
)
print(f"{mae=}\n{rmse=}\n{good_pixel_rate=}")

cv.imwrite("out.png", read_disp.astype(np.uint16) * 16, [cv.IMWRITE_PNG_COMPRESSION, 3])
