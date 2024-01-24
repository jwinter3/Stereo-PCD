import cv2 as cv
import numpy as np
import numpy.typing as npt


from stereo_pcd.datasets.middlebury_dataset import MiddleburyDataset
from stereo_pcd.pointclouds import (
    pseudo_points_color,
    save_txt,
)
from stereo_pcd.stereo_match import MultiThreadsMatcher


def get_nw_disp(imgL, imgR) -> npt.NDArray:
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayL = np.float32(grayL)
    harrisL = cv.cornerHarris(grayL, 2, 3, 0.04)

    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    grayR = np.float32(grayR)
    harrisR = cv.cornerHarris(grayR, 2, 3, 0.04)

    matcher = MultiThreadsMatcher(
        imgL, imgR, harrisL, harrisR, 0.5, 707, 0.5, 0.02, -0.015, -0.008, 0.0001, 2
    )
    return matcher.match()


if __name__ == "__main__":
    dataset = MiddleburyDataset("...")

    sample = dataset.read_sample("artroom1", colored=True)
    disparity = get_nw_disp(sample.left_img, sample.right_img)

    cv.imwrite("nw_artroom1.png", disparity * 256)

    xyzrgb = pseudo_points_color(sample, disparity, min_disp=25)
    save_txt(xyzrgb, "artroom1.txt")
