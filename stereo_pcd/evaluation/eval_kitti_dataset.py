import cv2 as cv
import numpy as np
import numpy.typing as npt
import pathlib


def read_kitti_result(
    sample: str,
    result_dir: pathlib.Path,
    gt_format: bool = False,
    disp_mult: float = 16,
) -> npt.NDArray:
    disp = cv.imread(
        (result_dir / f"{sample}_10.png").as_posix(), cv.IMREAD_UNCHANGED
    ).astype(np.float32)

    if gt_format:
        return disp / disp_mult  # type:ignore

    return disp  # type:ignore
