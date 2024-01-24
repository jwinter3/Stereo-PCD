import cv2 as cv
import numpy as np

from stereo_pcd.datasets.kitti_det_dataset import KittiDetDataset

from stereo_pcd.pointclouds import (
    combine_pcds,
    remove_attributes,
    pseudo_points,
    save_binary,
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

    xyz = remove_attributes(sample.pcd)
    pseudo_xyz = pseudo_points(sample, disparity)

    combined_pcd = combine_pcds([xyz, pseudo_xyz])

    save_binary(combined_pcd, "000006_lidar+stereo.bin")
