# Stereo-PCD
Python package for combining data from stereo cameras and lidars.

The package contains several modules and subpackages.  One of the modules (`stereo_match`) has been implemented
in C++ and the other modules and packages in Python.

## Packages and tools required to install the `stereo-pcd` package
- gcc
- clang-tidy
- CMake
- Make
- Boost Python
- Boost Numpy
- PythonLibs3.8
- OpenCV (optionally)

For more information on their installation, see [cpp/README.md](cpp/README.md).

# C++
Instructions for building the project can be found in [cpp/README.md](cpp/README.md).

Building the project will place the stereo_match module (`.so` file) in the package tree - under the path `stereo_pcd/stereo_match.so`.
# Python
The Python language package has been prepared in such a way that it can be installed using the pip tool.

Due to the compilation of a module written in C++ during the installation of the package, it is required, to have the same packages and tools available during installation as when building a C++ project.

Installation of the package requires the following commands to be called in the project root directory:

```bash
python3.8 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install .
```

# Tests

The unit tests are located in the [test](test) directory.
Running the tests requires the following commands to be executed in the root directory of the repository:
```bash
    pip install pytest
    export PYTHONPATH=$(pwd)
    pytest
```
# Examples

Examples of code using the library can be found in the directory [examples](examples).

# Docs

How the documentation is generated is described in  [docs](docs/README.md).

# Datasets
Websites from which library-supported datasets can be downloaded:
- 2021 Mobile stereo datasets with ground truth: https://vision.middlebury.edu/stereo/data/scenes2021/
- KITTI Stereo 2012: https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo
- KITTI 3D Object Detection: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

# Author
- Jakub Winter
