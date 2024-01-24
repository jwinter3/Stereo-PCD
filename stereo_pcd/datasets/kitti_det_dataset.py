import glob
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Union, Tuple

from stereo_pcd.datasets.dataset import Dataset, SampleWithLidar
from stereo_pcd.geometry import calc_baseline, extend_rotation_matrix


class KittiDetDataset(Dataset):
    def __init__(self, root_path: Union[str, Path]) -> None:
        super().__init__(root_path)
        self.samples = sorted(
            [
                x.split("/")[-1].split(".")[0]
                for x in glob.glob((self.root_path / "image_2/*.png").as_posix())
            ]
        )

    def _path_to_left_img(self, sample: str) -> Path:  # type: ignore
        return self.root_path / "image_2" / f"{sample}.png"

    def _path_to_right_img(self, sample: str) -> Path:  # type: ignore
        return self.root_path / "image_3" / f"{sample}.png"

    def _path_to_lidar(self, sample: str) -> Path:
        return self.root_path / "velodyne" / f"{sample}.bin"

    def _path_to_calib_file(self, sample: str) -> Path:
        return self.root_path / "calib" / f"{sample}.txt"

    def read_image(
        self, sample: str, left: bool = True, colored: bool = False
    ) -> npt.NDArray:
        im_path = (
            self._path_to_left_img(sample) if left else self._path_to_right_img(sample)
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

        lines_splitted = [line.split(":") for line in lines if line.rstrip() != ""]
        matrixes = {
            line[0]: np.fromstring(line[1], dtype=np.float32, sep=" ").reshape(3, -1)
            for line in lines_splitted
        }

        matrixes["Tr_velo_to_cam"] = np.insert(
            matrixes["Tr_velo_to_cam"], 3, values=[0, 0, 0, 1], axis=0
        )

        focal = matrixes[camera][0][0]
        velo_to_cam = (
            extend_rotation_matrix(matrixes["R0_rect"]) @ matrixes["Tr_velo_to_cam"]
        )

        baseline = calc_baseline(
            matrixes[camera] @ velo_to_cam, matrixes[second_camera] @ velo_to_cam
        )

        return (matrixes[camera] @ velo_to_cam, focal, baseline)

    def read_lidar(self, sample: str) -> npt.NDArray:
        pcd_path = self._path_to_lidar(sample)
        if not pcd_path.is_file():
            raise ValueError(f"Invalid path to lidar point cloud (sample: {sample})")

        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)

        return pcd

    def read_sample(self, sample_name: str, colored: bool = False) -> SampleWithLidar:
        sample = super().read_sample(sample_name, colored)
        pcd = self.read_lidar(sample_name)

        return SampleWithLidar(
            pcd,
            sample.left_img,
            sample.right_img,
            sample.left_color_img,
            sample.projection,
            sample.baseline,
            sample.focal,
        )

    def __getitem__(self, index: int) -> Tuple[str, SampleWithLidar]:
        return super().__getitem__(index)  # type: ignore
