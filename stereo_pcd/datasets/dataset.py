from abc import ABC, abstractmethod
import cv2 as cv
import numpy.typing as npt
from pathlib import Path
from typing import Union, Tuple, List, Optional, no_type_check


class Sample:
    def __init__(
        self,
        left_img: npt.NDArray,
        right_img: npt.NDArray,
        left_color_img: npt.NDArray,
        projection: npt.NDArray,
        baseline: float,
        focal: float,
    ) -> None:
        self.left_img = left_img
        self.right_img = right_img
        self.left_color_img = left_color_img
        self.projection = projection
        self.baseline = baseline
        self.focal = focal


class StereoSample(Sample):
    def __init__(
        self,
        gt_disp: npt.NDArray,
        left_img: npt.NDArray,
        right_img: npt.NDArray,
        left_color_img: npt.NDArray,
        projection: npt.NDArray,
        baseline: float,
        focal: float,
    ) -> None:
        super().__init__(
            left_img, right_img, left_color_img, projection, baseline, focal
        )
        self.gt_disp = gt_disp


class SampleWithLidar(Sample):
    def __init__(
        self,
        pcd: npt.NDArray,
        left_img: npt.NDArray,
        right_img: npt.NDArray,
        left_color_img: npt.NDArray,
        projection: npt.NDArray,
        baseline: float,
        focal: float,
    ) -> None:
        super().__init__(
            left_img, right_img, left_color_img, projection, baseline, focal
        )
        self.pcd = pcd


class Dataset(ABC):
    def __init__(self, root_path: Union[str, Path]) -> None:
        self.root_path = Path(root_path)
        self.samples: List[str] = []
        self.index = 0

        if not self.root_path.is_dir():
            raise ValueError("Invalid path to directory with dataset")

    def _read_image(self, im_path: Path, colored: bool) -> npt.NDArray:
        if not im_path.is_file():
            raise ValueError(f"File {im_path.as_posix()} does not exists")

        image = (
            cv.imread(im_path.as_posix(), cv.IMREAD_COLOR)
            if colored
            else cv.imread(im_path.as_posix(), cv.IMREAD_GRAYSCALE)
        )

        if image is None:
            raise ValueError(f"Cannot read file {im_path.as_posix()}")

        return image  # type:ignore

    @abstractmethod
    def _path_to_left_img(self, sample_path: Path) -> Path:
        pass

    @abstractmethod
    def _path_to_right_img(self, sample_path: Path) -> Path:
        pass

    def read_sample(self, sample_name: str, colored: bool = False) -> Sample:
        left = self.read_image(sample_name, left=True, colored=colored)
        right = self.read_image(sample_name, left=False, colored=colored)
        left_color = (
            left if colored else self.read_image(sample_name, left=True, colored=True)
        )

        projection, baseline, focal = self.read_calibration(sample_name)

        return Sample(left, right, left_color, projection, focal, baseline)

    @abstractmethod
    def read_image(
        self, sample: str, left: bool = True, colored: bool = False
    ) -> npt.NDArray:
        pass

    @abstractmethod
    def read_calibration(self, sample: str) -> Tuple[npt.NDArray, float, float]:
        pass

    def __getitem__(self, index: int) -> Tuple[str, Sample]:
        sample_name = self.samples[index]
        sample = self.read_sample(sample_name, colored=True)

        return sample_name, sample

    def __next__(self) -> Tuple[str, Sample]:
        try:
            next_sample = self.__getitem__(self.index)
            self.index += 1
            return next_sample
        except IndexError:
            raise StopIteration

    @no_type_check
    def __iter__(self):
        self.index = 0
        return self


class StereoDataset(Dataset, ABC):
    def __init__(self, root_path: Union[str, Path]) -> None:
        super().__init__(root_path)
        self.invalid_value: Optional[float] = None

    @abstractmethod
    def read_gt_disparity(self, sample: str) -> npt.NDArray:
        pass

    def read_sample(self, sample_name: str, colored: bool = False) -> StereoSample:
        sample = super().read_sample(sample_name, colored)
        gt_disp = self.read_gt_disparity(sample_name)

        return StereoSample(
            gt_disp,
            sample.left_img,
            sample.right_img,
            sample.left_color_img,
            sample.projection,
            sample.baseline,
            sample.focal,
        )

    def __getitem__(self, index: int) -> Tuple[str, StereoSample]:
        return super().__getitem__(index)  # type: ignore
