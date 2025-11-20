# Diffused Fields

**Core package for diffusion PDE (i.e., heat equation) based methods on geometric manifolds.**

This package provides implementations of both traditional diffusion solvers and walk-on-spheres methods on point clouds for various data types (scalars, vectors, quaternions).

This is supplementary material for the paper **"Object-centric Task Representation and Transfer using Diffused Orientation Fields"**.

## Package Ecosystem

This is the **core package** that provides fundamental diffusion algorithms and geometric manifold operations. For robot manipulation applications, see the companion package:

- **[diffused_fields_robotics](https://github.com/idiap/diffused_fields_robotics)** - Robot manipulation applications including task representations, motion generation, and control (depends on this package)

## Features

- **Diffusion on Manifolds**: Scalar, vector, and quaternion diffusion on point clouds 
- **Walk-on-Spheres Methods**: Monte Carlo-based diffusion solvers for point clouds and various geometric primitives
- **Visualization Tools**: Interactive 3D visualization with Polyscope

## Installation

### Using Python 3.12 virtual environment (recommended)

```bash
# Create a virtual environment named 'df' with Python 3.12
python3.12 -m venv df

# Activate the virtual environment
source df/bin/activate

# Install the package in editable mode
pip install -e .
```

### Using conda (recommended for Open3D)

```bash
conda install open3d
pip install -e .
```


## Repository Structure

```
diffused_fields/
├── src/diffused_fields/     # Main package source code
│   ├── core/                # Core algorithms and data structures
│   ├── diffusion/           # Diffusion solvers (scalar, vector, quaternion)
│   ├── manifold/            # Manifold operations and geometry
│   ├── utils/               # Utility functions
│   ├── visualization/       # Visualization tools
│   └── baselines/           # Baseline methods for comparison
├── scripts/                 # Example scripts and demonstrations
├── data/                    # Sample point clouds and meshes
│   ├── pointclouds/         # .ply and .pcd files
│   └── meshes/              # .obj and .stl files
└── config/                  # Configuration files
```

## Paper and Citation

If you use this package in your research, please cite:

```bibtex
@online{bilalogluTactileErgodicControl2024,
  title = {Tactile {{Ergodic Control Using Diffusion}} and {{Geometric Algebra}}},
  author = {Bilaloglu, Cem and Löw, Tobias and Calinon, Sylvain},
  date = {2024-02-07},
  eprint = {2402.04862},
  eprinttype = {arxiv},
  eprintclass = {cs},
  url = {http://arxiv.org/abs/2402.04862}
}
```

## Data


## Reproducing Paper Results

All simulation data and plots from the paper can be generated using the scripts in this repository. 

## Dependencies

To compute the discrete Laplacian on point clouds (and also meshes if you want)
robust_laplacian: https://github.com/nmwsharp/robust-laplacians-py
```
@article{Sharp:2020:LNT,
  author={Nicholas Sharp and Keenan Crane},
  title={{A Laplacian for Nonmanifold Triangle Meshes}},
  journal={Computer Graphics Forum (SGP)},
  volume={39},
  number={5},
  year={2020}
}
```

For basic point cloud operations (another library can be easily used instead):
open3d: https://www.open3d.org

For visualizations:
plotly: https://polyscope.run/py/

For sparse matrix operations:
scipy: https://scipy.org

For linear algebra operations:
numpy: https://numpy.org


This code is maintained by Cem Bilaloglu and licensed under the MIT License.

Copyright (c) 2025 Idiap Research Institute contact@idiap.ch