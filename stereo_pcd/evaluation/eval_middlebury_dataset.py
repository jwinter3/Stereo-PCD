import cv2 as cv
import numpy as np
import numpy.typing as npt
import pathlib

from stereo_pcd.utils.read_pfm import read_pfm, decode_pfm


def read_middlebury_result(
    sample: str,
    result_dir: pathlib.Path,
    gt_format: bool = False,
    disp_mult: float = 16,
) -> npt.NDArray:
    if gt_format:
        with open(result_dir / f"{sample}.pfm", "rb") as fp:
            header, data = read_pfm(fp)

        return decode_pfm(header, data)

    disp = (
        cv.imread(
            (result_dir / f"{sample}.png").as_posix(), cv.IMREAD_UNCHANGED
        ).astype(np.float32)
        / disp_mult
    )
    disp[disp == 0] = float("inf")

    return disp  # type:ignore
