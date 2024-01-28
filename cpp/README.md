# C++ project

## Required tools
- gcc
- clang-tidy
- cmake
- make

In Ubuntu, they can be installed with the command:
```bash
sudo apt install gcc make cmake clang-tidy
```

## Required libraries

- Boost Python (libboost-all-dev)
- Boost Numpy (libboost-all-dev)
- PythonLibs3.8 (python3.8-dev)

In Ubuntu, they can be installed with the command:
```bash
sudo apt install libboost-all-dev python3.8-dev
```

In addition, it is recommended that the OpenCV library is also available - instructions on how to install it are available on the website: https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/.
Otherwise, it will be downloaded and complicated when
building the project, but it will significantly increase the compilation time.

The build scripts additionally copy shared objects, which are Python modules, to the corresponding positions in the Python package tree (`stereo_pcd/stereo_match.so`).

## Building the project

In order to build the project you need to have the packages given earlier installed and execute the following commands in the `cpp` directory:

```bash
mkdir build
cd build
cmake ..
make
```

The compilation results will appear in the `build` directory and its subdirectories, the executable file will be placed in the `app` subdirectory, the static C++ library in the `lib` subdirectory and the Python module (shared library) in the `module` subdirectory.

## How to run the executable file?
Together with the static C++ library and the Python module, an executable file will be compiled. It can be run by specifying the appropriate parameters:
``` 
Usage: path/to/left_img path/to/right_img path/to/output [start_gap_cost] [continue_gap_cost] [match_reward] [corner_ths] [max_disp] [num_threads] [output_multiplier]
```
Example of how to run if you are in the `build` directory:
```
./app/StereoMatchApp left.png right_img.out output.out -0.015 -0.008 0.02 0.0001 200 8 16
```

It takes paths to stereovision images and saves the calculated disparity map.