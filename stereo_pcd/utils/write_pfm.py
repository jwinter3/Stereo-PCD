import io
from typing import Union
import pathlib

import numpy as np
import numpy.typing as npt


def save_array_as_pfm(
    output_path: Union[str, pathlib.Path], array: npt.NDArray[np.float32]
) -> None:
    output_path = pathlib.Path(output_path)

    with open(output_path, "wb") as fp:
        write_pfm(fp, array)


def write_pfm(fp: io.BufferedWriter, array: npt.NDArray[np.float32]) -> None:
    shape = array.shape
    color = False
    if len(shape) == 3:
        if shape[2] not in [1, 3]:
            raise ValueError(
                "Image should contain one channel (mono) or three channels (color)"
            )
        color = True if shape[2] == 3 else False
    elif len(shape) != 2:
        raise ValueError("Invalid shape of the array")
    write_header(fp, shape[1], shape[0], colored=color)
    write_data(fp, array)


def write_header(
    fp: io.BufferedWriter,
    width: int,
    height: int,
    scale: float = 1,
    colored: bool = False,
    is_big_endian: bool = False,
) -> None:
    if scale <= 0 or scale > 1:
        raise ValueError("Scale does not belong to the range (0, 1]")
    if not is_big_endian:
        scale = -scale

    fp.write(b"PF\n" if colored else b"Pf\n")
    fp.write(f"{width} {height}\n".encode("ascii"))
    fp.write(f"{scale}\n".encode("ascii"))


def write_data(fp: io.BufferedWriter, array: npt.NDArray[np.float32]) -> None:
    fp.write(np.flip(array, axis=0).tobytes())


if __name__ == "__main__":
    a = np.array([[0, 1], [2, 3]], dtype=np.float32)
    with open("test.pfm", "wb") as fp:
        write_header(fp, 1080, 1920)
        write_data(fp, a)
