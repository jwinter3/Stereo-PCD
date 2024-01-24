from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt


def stat_pixels_to_percentages(
    all_pixels: int,
    valid_pixels: int,
    no_disp_pixels: int,
    no_gt_pixels: int,
    sum_error: float,
    sum_sq_error: float,
    good_pixels: Dict[float, float],
    eval_only_valid_pixel: bool = True,
) -> Tuple[float, float, Dict[float, float], float, float]:
    pixels = valid_pixels if eval_only_valid_pixel else (valid_pixels + no_disp_pixels)
    for ths in good_pixels.keys():
        good_pixels[ths] /= pixels

    return (
        sum_error / valid_pixels,
        np.sqrt(sum_sq_error / valid_pixels),
        good_pixels,
        no_gt_pixels / all_pixels,
        no_disp_pixels / (valid_pixels + no_disp_pixels),
    )


def calc_stats_pixels(
    calculated_disparity: npt.NDArray[np.float32],
    gt_disparity: npt.NDArray[np.float32],
    good_px_ths: int = 5,
    invalid_value: Optional[float] = 0,
) -> Tuple[int, int, int, int, int, int, Dict[float, int]]:
    sum_sq_error, sum_error, valid_pixels = 0, 0, 0
    no_gt_pixels, no_disp_pixels = 0, 0

    thresholds = [x / 2 for x in range(2 * good_px_ths + 1)]
    good_pixels = {ths: 0 for ths in thresholds}

    for row_gt, row_disp in zip(gt_disparity, calculated_disparity):
        for pixel_gt, pixel_disp in zip(row_gt, row_disp):
            if pixel_gt == invalid_value:
                no_gt_pixels += 1
                continue
            if pixel_disp == invalid_value:
                no_disp_pixels += 1
                continue

            valid_pixels += 1
            error = abs(pixel_disp - pixel_gt)

            for ths in thresholds:
                if error <= ths:
                    good_pixels[ths] += 1

            sum_error += error
            sum_sq_error += error**2

    return (
        gt_disparity.shape[0] * gt_disparity.shape[1],
        valid_pixels,
        no_disp_pixels,
        no_gt_pixels,
        sum_error,
        sum_sq_error,
        good_pixels,
    )
