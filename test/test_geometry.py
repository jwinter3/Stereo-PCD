from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from stereo_pcd.geometry import (
    calc_3d_point_from_pixels_from_2_cam,
    calc_focal,
    calc_projection_matrix,
    pixel_to_world,
    world_to_pixel,
)


@pytest.mark.parametrize(
    "fov, im_height, im_width, camera_rotation, camera_translation, pixel, depth, expected_point",
    [
        (
            90,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([9, 12, -23], dtype=np.float32),
            (587, 424),
            4.5,
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([10, 12, -23], dtype=np.float32),
            (676, 424),
            4.5,
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (0, 50),
            1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, -1], dtype=np.float32),
            (100, 50),
            -1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32),
            (0, 100),
            1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, -1, -1], dtype=np.float32),
            (100, 100),
            -1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            180,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (50, 50),
            1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            120,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (50 - 50 * np.tan(np.pi / 6), 50),
            1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            120,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (50 + 50 * np.tan(np.pi / 6), 50),
            1,
            np.array([0, 0, 0], dtype=np.float32),
        ),
    ],
)
def test_pixel_to_world(
    fov: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    pixel: Tuple[int, int],
    depth: float,
    expected_point: npt.NDArray[np.float32],
) -> None:
    projection = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )

    assert np.isclose(
        pixel_to_world(projection, *pixel, depth), expected_point, rtol=0.005
    ).all()


@pytest.mark.parametrize(
    "fov, im_height, im_width, camera_rotation, camera_translation, point, expected_pixel",
    [
        (
            90,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([9, 12, -23], dtype=np.float32),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
            (587, 424),
        ),
        (
            90,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([10, 12, -23], dtype=np.float32),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
            (676, 424),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (0, 50),
        ),
        (90, 100, 100, (0, 180, 0), [0, 0, 0], [1, 0, -1], (0, 50)),
        (90, 100, 100, (0, 180, 0), [1, 0, -1], [0, 0, 0], (0, 50)),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (100, 50),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (0, 100),
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (100, 100),
        ),
        (
            180,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (50, 50),
        ),
        (
            120,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (50 - 50 * np.tan(np.pi / 6), 50),
        ),
        (
            120,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            (50 + 50 * np.tan(np.pi / 6), 50),
        ),
    ],
)
def test_world_to_pixel(
    fov: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    point: npt.NDArray[np.float32],
    expected_pixel: Tuple[float, float],
) -> None:
    projection = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )

    pixel = world_to_pixel(projection, *point)
    assert pixel is not None

    assert np.isclose(pixel, expected_pixel, atol=1).all()


@pytest.mark.parametrize(
    "fov, im_height, im_width, camera_rotation, camera_translation, point",
    [
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, -1], dtype=np.float32),
            [0, 0, 0],
        ),
        (
            90,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, -1, -1], dtype=np.float32),
            [0, 0, 0],
        ),
    ],
)
def test_world_to_pixel_point_behind_camera(
    fov: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    point: npt.NDArray[np.float32],
) -> None:
    projection = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )
    assert world_to_pixel(projection, *point) is None


@pytest.mark.parametrize(
    "fov, distance, im_height, im_width, camera_rotation, camera_translation, pixel_cam1, pixel_cam2, expected_point",
    [
        (
            90,
            1,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([10, 12, -23], dtype=np.float32),
            (676, 424),
            (587, 424),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            -1,
            600,
            800,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([9, 12, -23], dtype=np.float32),
            (587, 424),
            (676, 424),
            np.array([-6.9, -10.6, 27.5], dtype=np.float32),
        ),
        (
            90,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (100, 50),
            (0, 50),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, -1], dtype=np.float32),
            (0, 50),
            (100, 50),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            (100, 100),
            (0, 100),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            90,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32),
            (0, 100),
            (100, 100),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            120,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (50 + 50 * np.tan(np.pi / 6), 50),
            (50 - 50 * np.tan(np.pi / 6), 50),
            np.array([0, 0, 0], dtype=np.float32),
        ),
        (
            120,
            -2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([-1, 0, 1], dtype=np.float32),
            (50 - 50 * np.tan(np.pi / 6), 50),
            (50 + 50 * np.tan(np.pi / 6), 50),
            np.array([0, 0, 0], dtype=np.float32),
        ),
    ],
)
def test_calc_3d_point_from_pixels_from_2_cam(
    fov: float,
    distance: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    pixel_cam1: Tuple[int, int],
    pixel_cam2: Tuple[int, int],
    expected_point: npt.NDArray[np.float32],
) -> None:
    projection = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )
    focal = calc_focal(fov, im_width)
    point_cam = calc_3d_point_from_pixels_from_2_cam(
        pixel_cam1, pixel_cam2, projection, focal, distance
    )

    assert np.isclose(point_cam, expected_point, rtol=0.005).all()


@pytest.mark.parametrize(
    "fov, distance, im_height, im_width, camera_rotation, camera_translation, pixel_cam1, pixel_cam2",
    [
        (
            180,
            2,
            100,
            100,
            np.array([0, 0, 0], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            (50, 50),
            (50, 50),
        )
    ],
)
def test_calc_3d_point_from_pixels_from_2_degenerate_cam(
    fov: float,
    distance: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
    pixel_cam1: Tuple[int, int],
    pixel_cam2: Tuple[int, int],
) -> None:
    projection = calc_projection_matrix(
        fov, im_height, im_width, camera_rotation, camera_translation
    )
    focal = calc_focal(fov, im_width)

    with pytest.raises(ValueError):
        calc_3d_point_from_pixels_from_2_cam(
            pixel_cam1, pixel_cam2, projection, focal, distance
        )
