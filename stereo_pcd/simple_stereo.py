import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional

from stereo_pcd.geometry import calc_3d_points_matrix


def is_black(pixel: List[int]) -> bool:
    return sum(pixel[:3]) == 0


def find_similar_color(
    pixel: Tuple[int, int],
    left_x: int,
    right_row: npt.NDArray[np.int32],
    d_min: int,
    d_max: int,
    im_width: int,
) -> Optional[int]:
    best_fit = None
    min_diff = float("inf")

    for i in range(left_x + d_max, left_x + d_min, -1):
        if i < im_width and not is_black(right_row[i]):
            diff = sum(abs(pixel[:3] - right_row[i][:3]))
            if diff < min_diff:
                min_diff = diff
                best_fit = i

    return best_fit - left_x if best_fit is not None else 0


def calc_disparity_map(
    left_array: npt.NDArray[np.uint8],
    right_array: npt.NDArray[np.uint8],
    min_x: int,
    max_x: int,
) -> npt.NDArray[np.int32]:
    disparity = np.zeros(left_array.shape[:-1], dtype=np.int32)

    for i, (left_row, right_row) in enumerate(zip(left_array, right_array)):
        for j, pixel in enumerate(left_row):
            if not is_black(pixel):
                disparity[i][j] = find_similar_color(
                    pixel, j, right_row, min_x, max_x, left_array.shape[1]
                )
    return disparity


def calc_stereo_points(
    left_array: npt.NDArray[np.uint8],
    right_array: npt.NDArray[np.uint8],
    projection_left: npt.NDArray[np.float32],
    focal: float,
    distance: float,
    x_min: int,
    x_max: int,
) -> npt.NDArray[np.float32]:
    disparity = calc_disparity_map(left_array, right_array, x_min, x_max)

    return calc_3d_points_matrix(disparity, projection_left, focal, distance, x_min)


def create_disparity_picture(
    disparity_map: npt.NDArray[np.int32],
) -> npt.NDArray[np.uint8]:
    disparity_bitmap = np.empty((*disparity_map.shape, 2), dtype=np.uint8)

    disparity_bitmap[:, :, 0] = disparity_map
    disparity_bitmap[:, :, 1].fill(255)

    return disparity_bitmap
