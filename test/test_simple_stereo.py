import numpy as np
import numpy.typing as npt
from typing import Tuple

import pytest

from stereo_pcd.geometry import (
    calc_focal,
    calc_projection_matrix,
)
from stereo_pcd.simple_stereo import calc_stereo_points


@pytest.mark.parametrize(
    "fov, distance, im_height, im_width, d_min, d_max, camera_rotation, "
    "camera_translation, pixel_cam1, pixel_cam2, expected_point",
    [
        (
            90,
            -1,
            600,
            800,
            -90,
            72,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([10, 12, -23], dtype=np.float32),
            (676, 424),
            (587, 424),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            1,
            600,
            800,
            72,
            90,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([9, 12, -23], dtype=np.float32),
            (587, 424),
            (676, 424),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            11,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            -2,
            10,
            10,
            -11,
            0,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (10, 5),
            (0, 5),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            11,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            11,
            np.array([180, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
            np.array([0, 0, 0], dtype=np.float32),
        ),
    ],
)
def test_calc_stereo_point_from_2_bitmap(
    fov: float,
    distance: float,
    im_height: int,
    im_width: int,
    d_min: int,
    d_max: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    pixel_cam1: Tuple[int, int],
    pixel_cam2: Tuple[int, int],
    expected_point: npt.NDArray[np.float32],
) -> None:
    focal = calc_focal(fov, im_width)

    left_picture = np.zeros((im_height + 1, im_width + 1, 4), np.uint8)
    right_picture = np.zeros((im_height + 1, im_width + 1, 4), np.uint8)
    left_picture[:, :, 3].fill(255)

    left_picture[pixel_cam1[1]][pixel_cam1[0]][0:3] = np.array([255, 255, 255])
    right_picture[pixel_cam2[1]][pixel_cam2[0]][0:3] = np.array([255, 255, 255])

    projection_cam1 = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )

    points_cam = calc_stereo_points(
        left_picture,
        right_picture,
        projection_cam1,
        focal,
        distance,
        d_min,
        d_max,
    )

    assert np.isclose(expected_point, points_cam.T, rtol=0.005).any()


@pytest.mark.parametrize(
    "fov, distance, im_height, im_width, d_min, d_max, camera_rotation, camera_translation, pixel_cam1, pixel_cam2",
    [
        (
            90,
            -1,
            600,
            800,
            -88,
            72,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([10, 12, -23], dtype=np.float32),
            (676, 424),
            (587, 424),
        ),
        (
            90,
            1,
            600,
            800,
            72,
            88,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([9, 12, -23], dtype=np.float32),
            (587, 424),
            (676, 424),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            9,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
        ),
        (
            90,
            -2,
            10,
            10,
            -10,
            0,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (10, 5),
            (0, 5),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            9,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
        ),
        (
            90,
            2,
            10,
            10,
            0,
            9,
            np.array([180, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (0, 5),
            (10, 5),
        ),
    ],
)
def test_calc_stereo_point_from_2_bitmap_no_points(
    fov: float,
    distance: float,
    im_height: int,
    im_width: int,
    d_min: int,
    d_max: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    pixel_cam1: Tuple[int, int],
    pixel_cam2: Tuple[int, int],
) -> None:
    focal = calc_focal(fov, im_width)

    left_picture = np.zeros((im_height + 1, im_width + 1, 4), np.uint8)
    right_picture = np.zeros((im_height + 1, im_width + 1, 4), np.uint8)
    left_picture[:, :, 3].fill(255)

    left_picture[pixel_cam1[1]][pixel_cam1[0]][0:3] = np.array([255, 255, 255])
    right_picture[pixel_cam2[1]][pixel_cam2[0]][0:3] = np.array([255, 255, 255])

    projection_cam1 = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )

    points_cam = calc_stereo_points(
        left_picture,
        right_picture,
        projection_cam1,
        focal,
        distance,
        d_min,
        d_max,
    )

    assert points_cam.shape == (3, (im_width + 1) * (im_height + 1))
    assert np.isclose(points_cam.T, -camera_translation).all()
