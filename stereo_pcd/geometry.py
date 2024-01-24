import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple


def deg_to_rad(deg: float) -> float:
    return 2 * np.pi * deg / 360.0


def calc_focal(fov: float, im_width: int) -> float:
    return float(im_width / (2.0 * np.tan(deg_to_rad(fov) / 2)))


def calc_intrinsic_matrix(
    fov: float, im_height: int, im_width: int
) -> npt.NDArray[np.float32]:
    focal = calc_focal(fov, im_width)

    intrinsic = np.identity(3, dtype=np.float32)
    # In this case Fx and Fy are the same since the pixel aspect ratio is 1
    intrinsic[0, 0] = intrinsic[1, 1] = focal
    intrinsic[0, 2] = im_width / 2.0
    intrinsic[1, 2] = im_height / 2.0

    return intrinsic


def calc_yaw_rotation(yaw: float) -> npt.NDArray[np.float32]:
    yaw_rad = deg_to_rad(yaw)
    rotation = np.identity(3, dtype=np.float32)
    rotation[0][0] = rotation[1][1] = np.cos(yaw_rad)
    rotation[0][1] = -np.sin(yaw_rad)
    rotation[1][0] = np.sin(yaw_rad)

    return rotation


def calc_pitch_rotation(pitch: float) -> npt.NDArray[np.float32]:
    pitch_rad = deg_to_rad(pitch)
    rotation = np.identity(3, dtype=np.float32)
    rotation[0][0] = rotation[2][2] = np.cos(pitch_rad)
    rotation[2][0] = -np.sin(pitch_rad)
    rotation[0][2] = np.sin(pitch_rad)

    return rotation


def calc_roll_rotation(roll: float) -> npt.NDArray[np.float32]:
    roll_rad = deg_to_rad(roll)
    rotation = np.identity(3, dtype=np.float32)
    rotation[1][1] = rotation[2][2] = np.cos(roll_rad)
    rotation[1][2] = -np.sin(roll_rad)
    rotation[2][1] = np.sin(roll_rad)

    return rotation


def calc_rotation_matrix(
    yaw: float, pitch: float, roll: float
) -> npt.NDArray[np.float32]:
    return (
        calc_yaw_rotation(yaw) @ calc_pitch_rotation(pitch) @ calc_roll_rotation(roll)
    )


def extend_rotation_matrix(rotation: npt.NDArray) -> npt.NDArray:
    ext_rotation = np.concatenate(
        (rotation, np.array([[0], [0], [0]], dtype=np.float32)), axis=1
    )
    ext_rotation = np.concatenate(
        (ext_rotation, np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0
    )

    return ext_rotation  # type:ignore


def calc_extrinsic_matrix(
    rotation: npt.NDArray[np.float32], translation: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    translation_matrix = np.identity(4, dtype=np.float32)
    translation_matrix[:-1, 3] = translation

    ext_rotation = np.concatenate(
        (rotation, np.array([[0], [0], [0]], dtype=np.float32)), axis=1
    )
    ext_rotation = np.concatenate(
        (ext_rotation, np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0
    )

    return ext_rotation @ translation_matrix


def calc_projection_matrix_from_intrinsic_and_extrinsic(
    intrinsic: npt.NDArray[np.float32], extrinsic: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    intrinsic = np.concatenate(
        (intrinsic, np.array([[0], [0], [0]], dtype=np.float32)), axis=1
    )
    return intrinsic @ extrinsic


def calc_projection_matrix(
    fov: float,
    im_height: int,
    im_width: int,
    camera_rotation: npt.NDArray[np.float32],
    camera_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    intrinsic = calc_intrinsic_matrix(fov, im_height, im_width)
    rotation = calc_rotation_matrix(*camera_rotation)
    extrinsic = calc_extrinsic_matrix(rotation, camera_translation)
    projection = calc_projection_matrix_from_intrinsic_and_extrinsic(
        intrinsic, extrinsic
    )
    return projection


def extend_projection_matrix(
    projection_matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return np.concatenate(
        (projection_matrix, np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0
    )


def calc_baseline(
    projection_cam1: npt.NDArray[np.float32], projection_cam2: npt.NDArray[np.float32]
) -> float:
    point_cam1 = np.linalg.solve(
        extend_projection_matrix(projection_cam1),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )
    point_cam2 = np.linalg.solve(
        extend_projection_matrix(projection_cam2),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )

    return float(np.linalg.norm(point_cam1 - point_cam2))


def pixel_with_depth(
    pixel_x: int, pixel_y: int, depth: float
) -> npt.NDArray[np.float32]:
    return np.asarray([pixel_x * depth, pixel_y * depth, depth, 1], dtype=np.float32)


def calc_pixel_with_depth(
    pixel_x: int,
    pixel_y: int,
    pixel_disparity: float,
    focal: float,
    distance: float,
) -> Optional[npt.NDArray[np.float32]]:
    depth = calc_pixel_depth(pixel_disparity, focal, distance)

    if depth < 0:
        return None

    return pixel_with_depth(pixel_x, pixel_y, depth)


def calc_pixel_depth(pixel_disparity: float, focal: float, distance: float) -> float:
    return focal * distance / pixel_disparity


def pixel_to_world(
    projection_matrix: npt.NDArray[np.float32],
    pixel_x: int,
    pixel_y: int,
    depth: float,
) -> npt.NDArray[np.float32]:
    camera_point = pixel_with_depth(pixel_x, pixel_y, depth)
    if projection_matrix.shape == (3, 4):
        return np.linalg.solve(
            extend_projection_matrix(projection_matrix), camera_point
        )[:-1]
    return np.linalg.solve(projection_matrix, camera_point)[:-1]


def world_to_pixel(
    projection_matrix: npt.NDArray[np.float32], x: float, y: float, z: float
) -> Optional[npt.NDArray[np.float32]]:
    point = np.asarray([x, y, z, 1], dtype=np.float32)
    camera_point = projection_matrix @ point
    if camera_point[-1] < 0:
        return None
    camera_point = camera_point / camera_point[-1]
    return camera_point[:-1]


def calc_3d_points(
    disparity_map: npt.NDArray[np.int32],
    projection: npt.NDArray[np.float32],
    focal: float,
    distance: float,
) -> List[npt.NDArray[np.float32]]:
    points_3d = []

    for i, row in enumerate(disparity_map):
        for j, pixel_disparity in enumerate(row):
            if pixel_disparity != 0:
                points_3d.append(
                    calc_3d_point_from_disparity(
                        (j, i), pixel_disparity, projection, focal, distance
                    )
                )
    return points_3d


def calc_3d_point_from_disparity(
    pixel: Tuple[int, int],
    disparity: int,
    projection_cam1: npt.NDArray[np.float32],
    focal: float,
    distance: float,
) -> npt.NDArray[np.float32]:
    if disparity == 0:
        raise ValueError("Pixel coordinates should be different")

    camera_depth = calc_pixel_depth(disparity, focal, distance)

    return pixel_to_world(projection_cam1, *pixel, camera_depth)


def calc_3d_points_matrix(
    disparity_map: npt.NDArray[np.int32],
    projection_cam1: npt.NDArray[np.float32],
    focal: float,
    distance: float,
    min_disp: float = 0,
) -> npt.NDArray[np.float32]:
    camera_points = []
    for i, row in enumerate(disparity_map):
        for j, pixel_disparity in enumerate(row):
            if pixel_disparity <= min_disp or pixel_disparity == 0:
                camera_point = calc_pixel_with_depth(
                    j, i, float("inf"), focal, distance
                )
            else:
                camera_point = calc_pixel_with_depth(
                    j, i, pixel_disparity, focal, distance
                )
            if camera_point is not None:
                camera_points.append(camera_point)

    if len(camera_points) == 0:
        return np.empty((3, 0), dtype=np.float32)

    camera_points_mat = np.asarray(camera_points, dtype=np.float32)

    return (
        np.linalg.inv(extend_projection_matrix(projection_cam1)) @ camera_points_mat.T
    )[:-1]


def calc_3d_point_from_pixels_from_2_cam(
    pixel_cam1: Tuple[int, int],
    pixel_cam2: Tuple[int, int],
    projection_cam1: npt.NDArray[np.float32],
    focal: float,
    distance: float,
) -> npt.NDArray[np.float32]:
    disparity = pixel_cam1[0] - pixel_cam2[0]

    return calc_3d_point_from_disparity(
        pixel_cam1, disparity, projection_cam1, focal, distance
    )
