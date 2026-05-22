import os
os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"

import torch
import torch.utils.cpp_extension
import numpy as np
import shutil

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

_CACHE_DIR = os.path.expanduser("~/.cache/torch_extensions/py310_cu124/point_filter_v2_ops")
if os.path.exists(_CACHE_DIR):
    shutil.rmtree(_CACHE_DIR)

_point_filter_v2_ops = torch.utils.cpp_extension.load_inline(
    name="point_filter_v2_ops",
    cpp_sources="""
    torch::Tensor multiview_consistency_v2(
        torch::Tensor points_3d,
        torch::Tensor depth_maps,
        torch::Tensor depth_conf,
        torch::Tensor extrinsics,
        torch::Tensor intrinsics,
        torch::Tensor neighbor_indices,
        torch::Tensor cam_centers,
        torch::Tensor gradient_maps,
        float consistency_thresh,
        float lambda_grad);
    torch::Tensor weighted_color_fusion_v2(
        torch::Tensor points_3d,
        torch::Tensor images,
        torch::Tensor extrinsics,
        torch::Tensor intrinsics,
        torch::Tensor cam_centers,
        torch::Tensor neighbor_indices,
        torch::Tensor consistency_score,
        float consistency_min);
    """,
    cuda_sources=[open(os.path.join(_SRC_DIR, "point_filter_cuda_v2.cu"), "r").read()],
    functions=["multiview_consistency_v2", "weighted_color_fusion_v2"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


def compute_camera_centers(extrinsics_np):
    """
    Compute world-space camera centers from extrinsics [S, 3, 4].
    Center = -R^T @ t
    """
    R = extrinsics_np[:, :3, :3]              # [S, 3, 3]
    t = extrinsics_np[:, :3, 3]               # [S, 3]
    centers = -np.einsum('sij,sj->si', R.transpose(0, 2, 1), t)  # [S, 3]
    return centers


def _compute_gradient_maps(images: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel gradient magnitude for each image, normalized to [0, 1].
    images: [S, H, W, 3] in [0, 255]
    returns: [S, H, W] float32 tensor
    """
    import torch.nn.functional as F

    S, H, W, _ = images.shape
    # RGB to grayscale
    gray = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
    gray = gray.unsqueeze(1)  # [S, 1, H, W]

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=images.device) / 8.0
    sobel_y = sobel_x.T
    kx = sobel_x.view(1, 1, 3, 3)
    ky = sobel_y.view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx.squeeze(1) ** 2 + gy.squeeze(1) ** 2)  # [S, H, W]

    # Normalize globally
    gmax = grad.max()
    if gmax > 0:
        grad = grad / gmax
    return grad


def compute_neighbor_indices(extrinsics_np, num_neighbors=4):
    """
    Select K best neighbor views per frame based on camera center proximity
    (spatial adjacency) and viewing direction similarity.

    Uses camera center distance as primary signal — frames with nearby
    camera positions have good overlap for geometric verification.
    """
    centers = compute_camera_centers(extrinsics_np)  # [S, 3]
    S = len(centers)

    # Pairwise distances between camera centers
    diff = centers[:, None, :] - centers[None, :, :]  # [S, S, 3]
    dists = np.sqrt((diff ** 2).sum(axis=-1))          # [S, S]

    # Exclude self (set distance to inf)
    np.fill_diagonal(dists, np.inf)

    # Top-K nearest cameras per frame
    neighbor_indices = np.argsort(dists, axis=1)[:, :num_neighbors]  # [S, K]
    return neighbor_indices.astype(np.int32)


class PointCloudFilterV2:

    def __init__(
        self,
        num_neighbors: int = 4,
        consistency_thresh: float = 0.05,
        target_points: int = 100000,
        consistency_min: float = 0.5,
        lambda_grad: float = 0.3,
    ):
        self.num_neighbors = num_neighbors
        self.consistency_thresh = consistency_thresh
        self.target_points = target_points
        self.consistency_min = consistency_min
        self.lambda_grad = lambda_grad

    def filter(
        self,
        points_3d: torch.Tensor,        # [S, H, W, 3]
        depth_maps: torch.Tensor,       # [S, H, W]
        depth_conf: torch.Tensor,       # [S, H, W]
        extrinsics: torch.Tensor,       # [S, 3, 4]
        intrinsics: torch.Tensor,       # [S, 3, 3]
        images: torch.Tensor,           # [S, H, W, 3]
    ) -> dict:
        """
        Run the full filtering pipeline with v2 kernels.
        """
        S, H, W, _ = points_3d.shape
        device = points_3d.device

        # ---- Precompute on CPU from numpy ----
        extr_np = extrinsics.cpu().numpy()

        # Camera centers [S, 3]
        cam_centers_np = compute_camera_centers(extr_np)
        cam_centers = torch.from_numpy(cam_centers_np).float().to(device)

        # Neighbor indices [S, K]
        nbr_np = compute_neighbor_indices(extr_np, self.num_neighbors)
        neighbor_indices = torch.from_numpy(nbr_np).to(device)

        # Compute image gradient maps for texture-awareness bonus
        gradient_maps = _compute_gradient_maps(images)  # [S, H, W] in [0, 1]

        print(f"  [V2] Neighbor selection: using camera-center proximity")
        print(f"  [V2] Consist thresh: {self.consistency_thresh}, lambda_grad: {self.lambda_grad}")

        # ---- Step 1: Multi-view consistency (v2 kernel) ----
        consistency_score = _point_filter_v2_ops.multiview_consistency_v2(
            points_3d.contiguous(),
            depth_maps.contiguous(),
            depth_conf.contiguous(),
            extrinsics.contiguous(),
            intrinsics.contiguous(),
            neighbor_indices,
            cam_centers,
            gradient_maps.contiguous(),
            self.consistency_thresh,
            self.lambda_grad,
        )  # [S*H*W]

        # ---- Step 2: Weighted color fusion (v2 kernel) ----
        fused_colors = _point_filter_v2_ops.weighted_color_fusion_v2(
            points_3d.contiguous(),
            images.contiguous(),
            extrinsics.contiguous(),
            intrinsics.contiguous(),
            cam_centers,
            neighbor_indices,
            consistency_score.contiguous(),
            self.consistency_min,
        )  # [S*H*W, 3]

        # ---- Step 3: Spatial stratified sampling (same as v1, CPU-side) ----
        depth_flat = depth_maps.reshape(-1)
        valid_depth = depth_flat > 0.01

        score = consistency_score.clone()
        score[~valid_depth] = 0.0

        num_valid = (score > self.consistency_min).sum().item()
        print(f"  [V2 Filter] Valid points after consistency: {num_valid} / {S*H*W}")

        flat_points = points_3d.reshape(-1, 3)
        flat_colors = fused_colors

        selected_mask = torch.zeros(S * H * W, dtype=torch.bool, device=device)

        if num_valid <= self.target_points:
            selected_mask = score > self.consistency_min
        else:
            score_2d = score.reshape(S, H, W)
            grid_h = max(8, H // 16)
            grid_w = max(8, W // 16)
            n_cells_h = H // grid_h
            n_cells_w = W // grid_w
            points_per_cell = max(1, self.target_points // (n_cells_h * n_cells_w))

            for gh in range(n_cells_h):
                for gw in range(n_cells_w):
                    h_start = gh * grid_h
                    h_end = min((gh + 1) * grid_h, H)
                    w_start = gw * grid_w
                    w_end = min((gw + 1) * grid_w, W)

                    cell_scores = score_2d[:, h_start:h_end, w_start:w_end]
                    cell_valid = cell_scores > self.consistency_min
                    n_cell_valid = cell_valid.sum().item()

                    if n_cell_valid == 0:
                        continue

                    cell_flat = cell_scores.reshape(-1)
                    k = min(points_per_cell, n_cell_valid)
                    if k > 0:
                        _, top_indices = torch.topk(cell_flat, k)
                        for top_idx in top_indices:
                            s_idx = (top_idx // (grid_h * grid_w)).item()
                            local = (top_idx % (grid_h * grid_w)).item()
                            li = local // grid_w
                            lj = local % grid_w
                            global_idx = s_idx * H * W + (h_start + li) * W + (w_start + lj)
                            selected_mask[global_idx] = True

            selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_indices.numel() > self.target_points:
                perm = torch.randperm(selected_indices.numel(), device=device)
                selected_indices = selected_indices[perm[:self.target_points]]
                selected_mask[:] = False
                selected_mask[selected_indices] = True

        selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
        n_selected = selected_indices.numel()
        print(f"  [V2 Filter] Selected points: {n_selected} (target: {self.target_points})")

        out_points = flat_points[selected_indices].cpu().numpy()
        out_colors = flat_colors[selected_indices].cpu().numpy()
        out_conf = score[selected_indices].cpu().numpy()

        source_indices = selected_indices.cpu().numpy().astype(np.int64)

        return {
            "points_3d": out_points,
            "points_rgb": out_colors,
            "points_conf": out_conf,
            "consistency_map": consistency_score.cpu().numpy(),
            "num_filtered": num_valid,
            "num_selected": n_selected,
            "source_indices": source_indices,
        }
