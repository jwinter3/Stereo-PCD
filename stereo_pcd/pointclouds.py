import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import List, Union

from stereo_pcd.datasets.dataset import Sample
from stereo_pcd.geometry import calc_3d_points_matrix


def pseudo_points(
    sample: Sample, disparity: npt.NDArray, min_disp: int = 5
) -> npt.NDArray:
    points = calc_3d_points_matrix(
        disparity, sample.projection, sample.focal, sample.baseline, min_disp
    ).T

    return points


def pseudo_points_color(
    sample: Sample, disparity: npt.NDArray, min_disp: int = 5
) -> npt.NDArray:
    if disparity.ndim == 2:
        im_width, im_height = disparity.shape
    elif disparity.ndim == 3:
        im_width, im_height, channels = disparity.shape
        if channels != 1:
            raise ValueError(
                "Disparity should be a two-dimensional image with one channel"
            )
        else:
            disparity = disparity.reshape((im_width, im_height))
    else:
        raise ValueError("Disparity should be a two-dimensional image with one channel")

    points = pseudo_points(sample, disparity, min_disp)

    img = sample.left_color_img.reshape(im_width * im_height, 3)
    img = np.flip(img, axis=1)

    xyzrgb = np.concatenate([points, img], axis=1)

    return xyzrgb  # type:ignore


def remove_attributes(pcd: npt.NDArray) -> npt.NDArray:
    return pcd[:, 0:3]


def color_lidar_points(
    pcd: npt.NDArray, projection: npt.NDArray, color_img: npt.NDArray
) -> npt.NDArray:
    pcd = np.insert(pcd[:, 0:3], 3, 1, axis=1).T

    camera_points = projection @ pcd
    camera_points[:2] /= camera_points[2, :]
    colors = np.zeros((camera_points.shape[1], 3), dtype=np.uint8)

    for i, pixel in enumerate(camera_points.T):
        if (
            0 < pixel[0] < color_img.shape[1]
            and 0 < pixel[1] < color_img.shape[0]
            and pixel[2] > 0
        ):
            colors[i] = color_img[int(pixel[1]), int(pixel[0])]

    xyzrgb = np.concatenate([pcd[:3].T, np.flip(colors, axis=1)], axis=1)

    return xyzrgb  # type:ignore


def combine_pcds(pcds: List[npt.NDArray]) -> npt.NDArray:
    return np.concatenate(pcds, axis=0)


def save_binary(pcd: npt.NDArray, path: Union[str, Path]) -> None:
    return pcd.astype(np.float32).tofile(path)


def save_txt(pcd: npt.NDArray, path: Union[str, Path]) -> None:
    return np.savetxt(path, pcd)
