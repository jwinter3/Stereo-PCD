import cv2 as cv
import numpy as np
from stereo_pcd.evaluation.eval_image import (
    calc_stats_pixels,
    stat_pixels_to_percentages,
)
from stereo_pcd.utils.write_pfm import save_array_as_pfm

from stereo_pcd.datasets.middlebury_dataset import MiddleburyDataset

dataset = MiddleburyDataset("...")
sample = dataset.read_sample("ladder1")

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

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16
disparity[disparity == 0] = float("inf")


mae, rmse, good_pixel_rate, *_ = stat_pixels_to_percentages(
    *calc_stats_pixels(
        disparity,
        gt,
        invalid_value=float("inf"),
    ),
    eval_only_valid_pixel=False,
)
print(f"{mae=}\n{rmse=}\n{good_pixel_rate=}")

save_array_as_pfm("out.pfm", disparity)
