import json
import pathlib

from stereo_pcd.datasets.kitti_stereo_dataset import KittiStereoDataset
from stereo_pcd.evaluation.eval_dataset import eval_dataset
from stereo_pcd.evaluation.eval_kitti_dataset import read_kitti_result

if __name__ == "__main__":
    dataroot = pathlib.Path("...")

    result_dir = pathlib.Path("...")

    dataset = KittiStereoDataset(dataroot)

    result_dict = eval_dataset(
        dataset,
        result_dir,
        read_kitti_result,
        {
            "gap_cost": -0.015,
            "continue_gap_cost": -0.005,
            "match_reward": 0.05,
            "edge_ths": 0.0001,
            "edge_detector": "harris",
        },
        is_result_in_gt_format=True,
        eval_only_valid_pixel=False,
    )

    with open(result_dir / "results.json", "w") as fp:
        json.dump(result_dict, fp, indent=2)
