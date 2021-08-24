# MeshHashingDTSDF

Accompanying code for our IROS 2019 paper
```
@InProceedings{DTSDF_IROS_2019,
  author    = {M. {Splietker} and S. {Behnke}},
  title     = {Directional {TSDF}: Modeling Surface Orientation for Coherent Meshes},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2019},
  pages     = {1727--1734}
}
```

**This project is deprecated in favor of our [new implementation](https://github.com/AIS-Bonn/DirectionalTSDF)**

This code is a fork of the original [MeshHashing](https://github.com/theNded/MeshHashing) by Dong et al.
Please make sure to cite our and their corresponding papers, if you use the code.

## Build Instructions:

Because the code used GPU accelerated code, it requires OpenCV to be build with `WITH_CUDA=ON`. You may need to set the `OpenCV_DIR` variable in CMakeLists.txt.

```
sudo apt install libglfw3-dev ros-melodic-sophus libglm-dev libcapnp-dev
git submodule update --init --recursive 
mkdir build
cd build
cmake ..
make 
```

You might experience some warnings or errors originating from compiling Eigen with CUDA. 
In that case use a more recent Eigen version (>= 3.3.9).

## Usage
```
cd bin
./reconstruction
```
The program is controlled by the config file `config/args.yml` and accompanying dataset config specified by the `dataset_type` field. E.g. `dataset_type: 3` corresponds to the config file `config/TUM3.yml` for datasets in the TUM fr3 format.
