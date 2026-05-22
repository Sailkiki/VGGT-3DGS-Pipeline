#!/usr/bin/env python3
"""
Render a camera flythrough along the actual training camera trajectory.
"""
import os, sys, argparse, math
code_dir = os.path.join(os.path.dirname(__file__), '..', 'gaussian-splatting')
sys.path.insert(0, code_dir)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from scipy.spatial.transform import Rotation, Slerp


class PipelineParamsDummy:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_views", type=int, default=300)
    parser.add_argument("--smooth_window", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    # Load gaussians
    gaussians = GaussianModel(3)
    ply = os.path.join(args.model_dir, "point_cloud", "iteration_30000", "point_cloud.ply")
    gaussians.load_ply(ply)

    # Load camera data
    import json
    with open(os.path.join(args.model_dir, "cameras.json")) as f:
        cam_data = json.load(f)

    H, W = cam_data[0]["height"], cam_data[0]["width"]
    fx, fy = cam_data[0]["fx"], cam_data[0]["fy"]
    fov_x = 2 * math.atan(W / (2 * fx))
    fov_y = 2 * math.atan(H / (2 * fy))

    # Extract positions and rotations, sort by position proximity
    positions = np.array([c["position"] for c in cam_data])
    rotations = np.array([c["rotation"] for c in cam_data])
    N = len(positions)

    # Sort by a path: use TSP-like nearest-neighbor ordering
    order = [0]
    remaining = set(range(1, N))
    while remaining:
        last = order[-1]
        next_idx = min(remaining, key=lambda i: np.linalg.norm(positions[i] - positions[last]))
        order.append(next_idx)
        remaining.remove(next_idx)

    positions = positions[order]
    rotations = rotations[order]

    # Smooth trajectory with moving average
    w = args.smooth_window
    pos_smooth = np.zeros_like(positions)
    for i in range(N):
        start = max(0, i-w//2)
        end = min(N, i+w//2+1)
        pos_smooth[i] = positions[start:end].mean(axis=0)

    # Interpolate along path to get num_views frames
    total_dist = np.sum(np.linalg.norm(np.diff(pos_smooth, axis=0), axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pos_smooth, axis=0), axis=1))])

    # SLERP rotations
    quats = Rotation.from_matrix(rotations).as_quat()
    quats_smooth = np.zeros_like(quats)
    for i in range(N):
        start = max(0, i-w//2)
        end = min(N, i+w//2+1)
        quats_smooth[i] = quats[start:end].mean(axis=0)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)

    interp_rots = Rotation.from_quat(quats_smooth)

    print(f"Path length: {total_dist:.2f}m, frames: {args.num_views}")

    for i in tqdm(range(args.num_views), desc="Rendering"):
        t = i / args.num_views
        target_dist = t * total_dist

        # Find segment
        seg = np.searchsorted(cum_dist, target_dist) - 1
        seg = max(0, min(N-2, seg))
        alpha = (target_dist - cum_dist[seg]) / max(1e-6, cum_dist[seg+1] - cum_dist[seg])
        alpha = max(0, min(1, alpha))

        # Interpolate position and rotation
        pos = pos_smooth[seg] * (1-alpha) + pos_smooth[seg+1] * alpha

        # SLERP between two rotations
        r0 = Rotation.from_quat(quats_smooth[seg])
        r1 = Rotation.from_quat(quats_smooth[seg+1])
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        R_interp = slerp(alpha).as_matrix()

        # Camera forward is -R[2] in camera convention
        R = R_interp
        T = -R @ pos

        dummy_img = Image.new('RGB', (W, H), (128, 128, 128))
        cam = Camera(
            resolution=(H, W), colmap_id=i, R=R.astype(np.float64), T=T.astype(np.float64),
            FoVx=fov_x, FoVy=fov_y, depth_params=None,
            image=dummy_img, invdepthmap=None,
            image_name=f"path_{i:04d}", uid=i,
        )

        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            result = render(cam, gaussians, PipelineParamsDummy(), bg)
            rendered = result["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        frame = (rendered * 255).astype(np.uint8)
        Image.fromarray(frame).save(os.path.join(args.out_dir, f"frame_{i:04d}.png"))

    print(f"\nDone. {args.num_views} frames → {args.out_dir}")
    print(f"Video: ffmpeg -framerate 30 -i {args.out_dir}/frame_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {args.out_dir}/flythrough.mp4")


if __name__ == "__main__":
    main()
