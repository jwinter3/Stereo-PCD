import numpy as np
import pathlib
from typing import Dict, Union, Callable, no_type_check

from stereo_pcd.datasets.dataset import StereoDataset
from stereo_pcd.evaluation.eval_image import (
    calc_stats_pixels,
    stat_pixels_to_percentages,
)


@no_type_check
def eval_dataset(
    dataset: StereoDataset,
    result_dir: Union[str, pathlib.Path],
    read_result_func: Callable,
    params: Dict[str, int],
    eval_only_valid_pixel: bool = True,
    is_result_in_gt_format: bool = True,
):
    result_dir = pathlib.Path(result_dir)
    result_dict = {"params": params}
    dataset_stats = [0, 0, 0, 0, 0, 0, {}]

    for sample_name, sample in dataset:
        gt = sample.gt_disp

        gt = gt.astype(np.float32)

        disparity = read_result_func(sample_name, result_dir, is_result_in_gt_format)

        sample_stats = calc_stats_pixels(
            disparity,
            gt,
            invalid_value=dataset.invalid_value,
        )

        dataset_stats[:-1] = [
            dataset_stat + sample_stat
            for dataset_stat, sample_stat in zip(dataset_stats[:-1], sample_stats[:-1])
        ]

        dataset_stats[-1] = {
            ths: dataset_stats[-1].get(ths, 0) + sample_stats[-1].get(ths, 0)
            for ths in sample_stats[-1].keys()
        }

        (
            mae,
            rmse,
            good_pixel_rate,
            no_gt_pixel,
            no_disp_pixel,
        ) = stat_pixels_to_percentages(
            *sample_stats,
            eval_only_valid_pixel=eval_only_valid_pixel,
        )

        result_dict[sample_name] = {
            "mae": mae,
            "rmse": rmse,
            "good_pixel_rate": good_pixel_rate,
            "mask": 1 - no_gt_pixel,
            "invalid": no_disp_pixel,
        }

    for stat_name, stat_func in [("average_macro", np.average), ("stdev", np.std)]:
        result_dict[stat_name] = {
            "mae": stat_func(
                [result_dict[sample_name]["mae"] for sample_name in dataset.samples]
            ),
            "rmse": stat_func(
                [result_dict[sample_name]["rmse"] for sample_name in dataset.samples]
            ),
            "good_pixel_rate": {
                ths: stat_func(
                    [
                        result_dict[sample_name]["good_pixel_rate"][ths]
                        for sample_name in dataset.samples
                    ]
                )
                for ths in result_dict[dataset.samples[0]]["good_pixel_rate"].keys()
            },
            "mask": stat_func(
                [result_dict[sample_name]["mask"] for sample in dataset.samples]
            ),
            "invalid": stat_func(
                [result_dict[sample_name]["invalid"] for sample in dataset.samples]
            ),
        }

    (
        mae,
        rmse,
        good_pixel_rate,
        no_gt_pixel,
        no_disp_pixel,
    ) = stat_pixels_to_percentages(
        *dataset_stats,
        eval_only_valid_pixel=eval_only_valid_pixel,
    )

    result_dict["average_micro"] = {
        "mae": mae,
        "rmse": rmse,
        "good_pixel_rate": good_pixel_rate,
        "mask": 1 - no_gt_pixel,
        "invalid": no_disp_pixel,
    }

    return result_dict
