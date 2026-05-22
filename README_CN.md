# VGGT → 3DGS pipeline

以 VGGT前馈位姿估计替代 COLMAP SfM，构建手机影像端到端三维重建管线。输入照片或视频，输出可自由视角渲染的 3D Gaussian Splatting 场景模型。

## 管线流程

```
照片/视频 → VGGT 前馈推理（位姿 + 深度 + 点云）→ CUDA 多视图筛选 → COLMAP 导出 → 3DGS 训练 → 新视角渲染
```

## 核心贡献

### 1. CUDA 多视图几何一致性点云筛选

VGGT 从深度图反投影产生约 3200 万候选三维点，默认随机采样 10 万点用于 3DGS 初始化。本模块用跨视角几何验证替代随机采样：

- 每个候选点投影至 4 个最近邻相机视角
- 对比几何三角化深度与 VGGT 预测深度，指数衰减软评分衡量一致性
- Partial bilinear 采样容忍边缘点部分角落无深度的情况
- Sobel 梯度图提供纹理感知加分，避免过度选取无纹理平坦区域
- 视线夹角余弦加权的多视角颜色融合
- 置信度自适应空间分层采样保证覆盖面均匀

### 2. 跨视角深度重投影约束

3DGS 训练时，从当前视角和最近邻视角分别渲染深度图。将邻居渲染深度经三维几何变换重投影至当前视角成像平面，与当前渲染深度逐像素比对 L1 误差。该自监督正则化不依赖 VGGT 深度精度，利用多视角几何关系约束模型几何自洽。

### 3. 基于贡献度排序的高斯剪枝

训练过程中跟踪每个高斯在所有训练视角下的最大渲染贡献。训练结束后按贡献度排序，删去末尾 30% 低贡献高斯，再经 3000 轮微调恢复。

## 多场景验证

| 场景 | 来源 | 帧数 | 分辨率 | Test PSNR |
|------|------|:----:|--------|:---------:|
| scene1 | 手机照片 | 120 | 720×1280 | 24.15 |
| scene2 | 手机照片 | 107 | 1920×2560 | 24.63 |
| video1 | 视频抽帧 | 112 | 720×1280 | 23.36 |
| video2 | 视频抽帧 | 116 | 720×1280 | 23.12 |

## 快速开始

### 环境要求

- Python 3.10+, PyTorch 2.x + CUDA 12.4
- pycolmap 3.10.0

### 安装

```bash
conda create -n vggt3d python=3.10
conda activate vggt3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pycolmap==3.10.0 open3d trimesh plyfile tqdm opencv-python

# VGGT 模型及权重
git clone https://github.com/facebookresearch/vggt.git
cd vggt && mkdir pretrained
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -P pretrained/

# 3D Gaussian Splatting
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
```

### 使用

```bash
# 第一步：VGGT 推理 + CUDA 筛选 → COLMAP 格式
python vggt/demo_colmap.py \
    --scene_dir data/my_scene \
    --model_path vggt/pretrained/model.pt \
    --use_cuda_filter --seed 42

# 第二步：3DGS 训练（15000 轮）
python gaussian-splatting/train.py \
    -s data/my_scene \
    -m output/my_scene \
    --eval --iterations 15000

# 第三步：渲染新视角
python gaussian-splatting/render.py -m output/my_scene --skip_train

# 第四步：评估指标
python gaussian-splatting/metrics.py -m output/my_scene
```

## 文件结构

```
├── scripts/
│   ├── point_filter_v2.py          # CUDA 筛选器 Python 封装
│   ├── point_filter_cuda_v2.cu     # CUDA kernel（一致性评分 + 颜色融合）
│   ├── export_vggt_depths.py       # 导出 VGGT 深度与置信度图
│   ├── test_cuda_filter.py         # CUDA 筛选器回归测试
│   ├── run_pipeline.py             # 端到端管线脚本
│   ├── render_trajectory.py        # 相机轨迹渲染
│   └── cinematic_flight.py         # 飞行路径生成
├── vggt/
│   ├── demo_colmap.py              # 修改：CUDA 筛选 + eval_split + XYF 修复
│   └── vggt/utils/load_fn.py       # 修改：白填充适配 VGGT 输入
├── gaussian-splatting/
│   ├── train.py                    # 修改：跨视角一致性 + 贡献度剪枝
│   ├── scene/gaussian_model.py     # 修改：贡献度累加 + 剪枝
│   ├── scene/__init__.py           # 修改：邻域相机预计算
│   ├── scene/cameras.py            # 修改：置信度加权深度 mask
│   ├── arguments/__init__.py       # 修改：一致性 + 剪枝参数
│   └── utils/camera_utils.py       # 修改：置信度图加载
└── README.md / README_CN.md
```

## 引用

本项目基于以下开源工作：
- **VGGT** — Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025
- **3D Gaussian Splatting** — Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023

## 许可证

本仓库的原创代码（scripts/ 及修改文件）以 MIT 许可证发布。VGGT 与 3DGS 组件保留其原始许可证。
