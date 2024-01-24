import numpy as np
import numpy.typing as npt

MAX_UINT8 = 255
RED = [0, 0, MAX_UINT8]


def error_image(
    disparity: npt.NDArray,
    gt: npt.NDArray,
    ths: float = 10,
    invalid_value: float = 0,
    min_green_value: int = 63,
) -> npt.NDArray:
    factor = MAX_UINT8 - min_green_value
    error_img = np.zeros((*gt.shape, 3), dtype=np.uint8)

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] == invalid_value or disparity[i][j] == invalid_value:
                continue

            error = abs(disparity[i][j] - gt[i][j])

            if error <= ths:
                color = int(((ths - error) / ths) * factor) + min_green_value
                error_img[i][j] = [0, color, 0]
            else:
                error_img[i][j] = RED

    return error_img
