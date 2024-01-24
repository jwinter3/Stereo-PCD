import io
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


def read_pfm(fp: io.BufferedReader) -> Tuple[List[str], bytes]:
    header = [fp.readline().decode("ascii").rstrip() for _ in range(3)]
    data = fp.read()
    return header, data


def decode_header(header: List[str]) -> Tuple[int, int, bool, bool]:
    if len(header) != 3:
        raise ValueError("Header should contain exactly 3 lines")

    pf_type, res, byte_order = header

    if pf_type not in ["PF", "Pf"]:
        raise ValueError("Unknown pfm file type")

    is_mono = pf_type == "Pf"
    is_big_endian = float(byte_order) > 0
    x_res, y_res = [int(x) for x in res.split()]

    return x_res, y_res, is_big_endian, is_mono


def decode_pfm(header: List[str], data: bytes) -> npt.NDArray:
    x_res, y_res, is_big_endian, is_mono = decode_header(header)

    if is_mono is False:
        raise NotImplementedError

    if len(data) != x_res * y_res * 4:
        raise ValueError("Invalid number of pixels")

    dt = np.dtype(">f4") if is_big_endian else np.dtype("<f4")

    pfm_array = np.frombuffer(data, dtype=dt).reshape(y_res, x_res)

    return np.flip(pfm_array, axis=0)
