# VGGT → 3DGS Pipeline

End-to-end 3D reconstruction pipeline that replaces COLMAP SfM with VGGT for camera pose estimation. Input smartphone photos or videos, output a renderable 3D Gaussian Splatting scene model.

## Pipeline

```
Photos/Videos → VGGT Inference (pose + depth + point cloud) → CUDA Multi-View Filter → COLMAP Export → 3DGS Training → Novel View Rendering
```

## Contributions

### 1. CUDA Multi-View Geometric Consistency Filter

VGGT back-projects depth maps into ~32M candidate 3D points, then randomly samples 100K for 3DGS initialization. This module replaces random sampling with cross-view geometric verification:

- Projects each candidate point to its 4 nearest-neighbor camera views
- Compares triangulated geometric depth against VGGT-predicted depth using soft exponential scoring
- Partial bilinear sampling tolerates edge points where some corners lack depth
- Sobel gradient maps provide a texture-awareness bonus, preventing over-selection of featureless flat regions
- View-angle weighted multi-view color fusion with cosine weighting
- Spatial stratified sampling ensures uniform coverage

### 2. Cross-View Depth Reprojection Constraint

During 3DGS training, renders depth maps from both the current viewpoint and its nearest neighbor. The neighbor depth is warped to the current view via 3D geometry, then compared against the current rendered depth with an L1 loss. This self-supervised regularizer enforces multi-view geometric consistency without relying on VGGT depth accuracy.

### 3. Contribution-Based Gaussian Pruning

Tracks per-Gaussian maximum rendering contribution across all training views. After training, Gaussians are sorted by contribution and the bottom 30% are pruned, followed by 3K iterations of fine-tuning.

## Multi-Scene Validation

| Scene  | Source       | Frames | Resolution | Test PSNR |
| ------ | ------------ | :----: | ---------- | :-------: |
| scene1 | Phone photos |  120   | 720×1280   |   24.15   |
| scene2 | Phone photos |  107   | 1920×2560  |   24.63   |
| video1 | Video frames |  112   | 720×1280   |   23.36   |
| video2 | Video frames |  116   | 720×1280   |   23.12   |

## Quick Start

### Requirements

- Python 3.10+, PyTorch 2.x + CUDA 12.4
- pycolmap 3.10.0

### Installation

```bash
conda create -n vggt3d python=3.10
conda activate vggt3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pycolmap==3.10.0 open3d trimesh plyfile tqdm opencv-python

# VGGT model and weights
git clone https://github.com/facebookresearch/vggt.git
cd vggt && mkdir pretrained
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -P pretrained/

# 3D Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
```

### Usage

```bash
# Step 1: VGGT inference + CUDA filter → COLMAP format
python vggt/demo_colmap.py \
    --scene_dir data/my_scene \
    --model_path vggt/pretrained/model.pt \
    --use_cuda_filter --seed 42

# Step 2: 3DGS training (15K iterations)
python gaussian-splatting/train.py \
    -s data/my_scene \
    -m output/my_scene \
    --eval --iterations 15000

# Step 3: Render novel views
python gaussian-splatting/render.py -m output/my_scene --skip_train

# Step 4: Evaluate metrics
python gaussian-splatting/metrics.py -m output/my_scene
```

## File Structure

```
├── scripts/
│   ├── point_filter_v2.py          # Python wrapper for CUDA filter
│   ├── point_filter_cuda_v2.cu     # CUDA kernels (consistency scoring + color fusion)
│   ├── export_vggt_depths.py       # Export VGGT depth & confidence maps
│   ├── test_cuda_filter.py         # CUDA filter regression tests
│   ├── run_pipeline.py             # End-to-end pipeline runner
│   ├── render_trajectory.py        # Camera trajectory rendering
│   └── cinematic_flight.py         # Flight path generation
├── vggt/
│   ├── demo_colmap.py              # Modified: CUDA filter + eval_split + XYF fix
│   └── vggt/utils/load_fn.py       # Modified: white padding for VGGT input
├── gaussian-splatting/
│   ├── train.py                    # Modified: cross-view consistency + contribution pruning
│   ├── scene/gaussian_model.py     # Modified: contribution accumulation + pruning
│   ├── scene/__init__.py           # Modified: neighbor camera precomputation
│   ├── scene/cameras.py            # Modified: confidence-weighted depth mask
│   ├── arguments/__init__.py       # Modified: consistency + pruning parameters
│   └── utils/camera_utils.py       # Modified: confidence map loading
└── README.md / README_CN.md
```

## Citation

This project builds on the following works. If you use this code, please also cite them:

**VGGT**

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny.  
*VGGT: Visual Geometry Grounded Transformer.* CVPR 2025.

- Paper: <https://arxiv.org/abs/2503.11651>
- Code: <https://github.com/facebookresearch/vggt>

**3D Gaussian Splatting**

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis.  
*3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM Trans. Graph. (SIGGRAPH 2023).

- DOI: <https://doi.org/10.1145/3592433>
- Code: <https://github.com/graphdeco-inria/gaussian-splatting>

## License

Original code in this repository (scripts/ and modified files) is released under the MIT License. VGGT and 3DGS components retain their original licenses.
