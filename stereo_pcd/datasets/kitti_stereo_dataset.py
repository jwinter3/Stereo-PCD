import cv2 as cv
import glob
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import numpy.typing as npt

from stereo_pcd.datasets.dataset import StereoDataset
from stereo_pcd.geometry import calc_baseline


class KittiStereoDataset(StereoDataset):
    def __init__(self, root_path: Union[str, Path]) -> None:
        super().__init__(root_path)
        self.samples = sorted(
            [
                x.split("/")[-1].split("_")[0]
                for x in glob.glob((self.root_path / "image_0/*_10.png").as_posix())
            ]
        )
        self.gt_mult = 256.0
        self.invalid_value = 0

    def _path_to_left_img(self, sample: str, colored: bool) -> Path:  # type: ignore
        return (
            self.root_path
            / f"{'colored' if colored else 'image'}_0"
            / f"{sample}_10.png"
        )

    def _path_to_right_img(self, sample: str, colored: bool) -> Path:  # type: ignore
        return (
            self.root_path
            / f"{'colored' if colored else 'image'}_1"
            / f"{sample}_10.png"
        )

    def _path_to_disp(self, sample: str, occluded: bool = False) -> Path:
        return (
            self.root_path
            / ("disp_occ" if occluded else "disp_noc")
            / f"{sample}_10.png"
        )

    def _path_to_calib_file(self, sample: str) -> Path:
        return self.root_path / "calib" / f"{sample}.txt"

    def read_image(
        self, sample: str, left: bool = True, colored: bool = False
    ) -> npt.NDArray:
        im_path = (
            self._path_to_left_img(sample, colored)
            if left
            else self._path_to_right_img(sample, colored)
        )

        return self._read_image(im_path, colored=colored)

    def read_calibration(
        self, sample: str, colored: bool = False, left: bool = True
    ) -> Tuple[npt.NDArray, float, float]:
        cameras = {
            (True, False): "P0",
            (False, False): "P1",
            (True, True): "P2",
            (False, True): "P3",
        }

        camera = cameras[left, colored]
        second_camera = cameras[not left, colored]

        calib_file = self._path_to_calib_file(sample)

        if not calib_file.is_file():
            raise ValueError(f"File {calib_file.as_posix()} does not exists")

        with open(calib_file, "r", encoding="ascii") as fp:
            lines = fp.readlines()

        lines_splitted = [line.split(":") for line in lines]
        projections = {line[0]: line[1] for line in lines_splitted}

        projection_matrix = np.fromstring(
            projections[camera], dtype=np.float32, sep=" "
        ).reshape(3, 4)

        second_projection_matrix = np.fromstring(
            projections[second_camera], dtype=np.float32, sep=" "
        ).reshape(3, 4)

        baseline = calc_baseline(projection_matrix, second_projection_matrix)
        focal = projection_matrix[0][0]

        return projection_matrix, baseline, focal

    def read_gt_disparity(self, sample: str) -> npt.NDArray:
        gt_disp_path = self._path_to_disp(sample)

        if not gt_disp_path.is_file():
            raise ValueError(f"File {gt_disp_path.as_posix()} does not exists")

        gt_disp: Optional[npt.NDArray] = cv.imread(
            gt_disp_path.as_posix(), cv.IMREAD_UNCHANGED
        )

        if gt_disp is None:
            raise ValueError(f"Cannot read file {gt_disp_path.as_posix()}")

        return gt_disp.astype(np.float32) / self.gt_mult
