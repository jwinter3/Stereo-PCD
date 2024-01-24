import cv2 as cv
import numpy as np

from stereo_pcd.datasets.kitti_stereo_dataset import KittiStereoDataset
from stereo_pcd.pointclouds import (
    pseudo_points_color,
    save_txt,
)


def get_sgbm_disp(imgL, imgR):
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

    return stereo.compute(imgL, imgR).astype(np.float32) / 16.0


if __name__ == "__main__":
    dataset = KittiStereoDataset("...")

    sample = dataset.read_sample("000006")
    disparity = get_sgbm_disp(sample.left_img, sample.right_img)

    cv.imwrite("000006_sgbm.png", disparity)

    xyzrgb = pseudo_points_color(sample, disparity)
    save_txt(xyzrgb, "000006.txt")
