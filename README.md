MeshHashing
------
Code (partly) for our
- ICRA paper (**Accepted**): *An Efficient Volumetric Mesh Representation for 
Real-time Scene Reconstruction using Spatial-hashing*
- ECCV paper (**Accepted**): *PSDF Fusion: Probabilistic Signed Distance Function for On-the-fly 3D Data Fusion and Scene Reconstruction*

**This project is deprecated. It is now being migrated to Open3D with CUDA components. Please refer to [this fork of Open3D](https://github.com/theNded/Open3D/tree/cuda).**

Installation instructions:
```
sudo apt install libglfw3-dev ros-melodic-sophus libglm-dev libcapnp-dev
git submodule update --init --recursive 
mkdir build
cd build
cmake ..
make 
```

You might experience some warnings or errors originating from compiling Eigen with cuda. 
In that case use a more recent Eigen version (>= 3.3.9).


The main program is called `reconstruction` and can be found under `bin/`.
The program is controlled by the config file `config/args.yml` and accompanying dataset config specified by
the `dataset_type` field. E.g. `dataset_type: 3` corresponds to the config file `config/TUM3.yml` for datasets in the TUM fr3 format. 