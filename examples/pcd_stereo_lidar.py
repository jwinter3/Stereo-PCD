import cv2 as cv
import numpy as np

from stereo_pcd.datasets.kitti_det_dataset import KittiDetDataset

from stereo_pcd.pointclouds import (
    pseudo_points_color,
    color_lidar_points,
    combine_pcds,
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
    dataset = KittiDetDataset("...")
    sample = dataset.read_sample("000006")
    disparity = get_sgbm_disp(sample.left_img, sample.right_img)

    xyzrgb = color_lidar_points(sample.pcd, sample.projection, sample.left_color_img)
    pseudo_xyzrgb = pseudo_points_color(sample, disparity)

    combined_pcd = combine_pcds([xyzrgb, pseudo_xyzrgb])

    save_txt(xyzrgb, "000006_colored_lidar.txt")
    save_txt(pseudo_xyzrgb, "000006_colored_stereo.txt")
    save_txt(combined_pcd, "000006_lidar+stereo.txt")
