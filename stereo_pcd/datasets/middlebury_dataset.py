import glob
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt

from stereo_pcd.datasets.dataset import StereoDataset
from stereo_pcd.geometry import (
    calc_rotation_matrix,
    calc_extrinsic_matrix,
    calc_projection_matrix_from_intrinsic_and_extrinsic,
)
from stereo_pcd.utils.read_pfm import read_pfm, decode_pfm


class MiddleburyDataset(StereoDataset):
    def __init__(self, root_path: Union[str, Path]) -> None:
        super().__init__(root_path)
        self.samples = sorted(
            [x.split("/")[-1] for x in glob.glob((self.root_path / "*").as_posix())]
        )
        self.invalid_value = float("inf")

    def _path_to_left_img(self, sample_path: Path) -> Path:
        return sample_path / "im0.png"

    def _path_to_right_img(self, sample_path: Path) -> Path:
        return sample_path / "im1.png"

    def _path_to_left_disp(self, sample_path: Path) -> Path:
        return sample_path / "disp0.pfm"

    def _path_to_right_disp(self, sample_path: Path) -> Path:
        return sample_path / "disp1.pfm"

    def _path_to_calib_file(self, sample_path: Path) -> Path:
        return sample_path / "calib.txt"

    def _calcutale_projection_matrix(
        self, intrinsic: npt.NDArray, baseline: float, is_left: bool = True
    ) -> npt.NDArray:
        rotation = calc_rotation_matrix(0, 0, 0)
        translation = np.array([0 if is_left else baseline, 0, 0], dtype=np.float32)
        extrinsic = calc_extrinsic_matrix(rotation, translation)

        return calc_projection_matrix_from_intrinsic_and_extrinsic(intrinsic, extrinsic)

    def get_sample_path(self, sample_name: str) -> Path:
        sample_path = self.root_path / sample_name
        if not sample_path.is_dir():
            raise ValueError(f"Sample {sample_name} not found")
        return sample_path

    def read_image(
        self, sample: str, left: bool = True, colored: bool = False
    ) -> npt.NDArray:
        sample_path = self.get_sample_path(sample)

        im_path = (
            self._path_to_left_img(sample_path)
            if left
            else self._path_to_right_img(sample_path)
        )

        return self._read_image(im_path, colored=colored)

    def read_calibration(
        self, sample: str, is_left: bool = True
    ) -> Tuple[npt.NDArray, float, float]:
        sample_path = self.get_sample_path(sample)
        calib_file = self._path_to_calib_file(sample_path)

        with open(calib_file, "r", encoding="ascii") as fp:
            lines = fp.readlines()

        lines_splitted = [line.split("=") for line in lines]
        params = {line[0]: line[1] for line in lines_splitted}

        baseline = float(params["baseline"]) / 1000  # mm to m

        cam0 = params["cam0"].lstrip("[").rstrip("]").replace(";", "")
        cam1 = params["cam1"].lstrip("[").rstrip("]").replace(";", "")

        intrinsic_cam0 = np.fromstring(cam0, dtype=np.float32, sep=" ").reshape(3, 3)
        intrinsic_cam1 = np.fromstring(cam1, dtype=np.float32, sep=" ").reshape(3, 3)

        intrinsic = intrinsic_cam0 if is_left else intrinsic_cam1
        focal = intrinsic[0][0]

        return (
            self._calcutale_projection_matrix(intrinsic, baseline, is_left),
            baseline,
            focal,
        )

    def read_gt_disparity(self, sample: str, left: bool = True) -> npt.NDArray:
        sample_path = self.get_sample_path(sample)
        gt_disp_path = (
            self._path_to_left_disp(sample_path)
            if left
            else self._path_to_right_disp(sample_path)
        )

        if not gt_disp_path.is_file():
            raise ValueError(f"File {gt_disp_path.as_posix()} does not exists")

        with open(gt_disp_path, "rb") as fp:
            header, data = read_pfm(fp)

        return decode_pfm(header, data)
