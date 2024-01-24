import json
import pathlib

from stereo_pcd.datasets.middlebury_dataset import MiddleburyDataset
from stereo_pcd.evaluation.eval_dataset import eval_dataset
from stereo_pcd.evaluation.eval_middlebury_dataset import read_middlebury_result


if __name__ == "__main__":
    result_dir = pathlib.Path("...")

    dataset = MiddleburyDataset("...")
    result_dict = eval_dataset(
        dataset,
        result_dir,
        read_middlebury_result,
        {
            "gap_cost": -0.015,
            "continue_gap_cost": -0.08,
            "match_reward": 0.02,
            "corner_ths": 0.0001,
            "edge_detector": "harris",
        },
        is_result_in_gt_format=False,
        eval_only_valid_pixel=False,
    )

    with open(result_dir / "results.json", "w") as fp:
        json.dump(result_dict, fp, indent=2)
