#!/usr/bin/env python3
"""定制飞行轨迹：远处螺旋推进 + 环绕 + 抬高"""
import os, sys, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gaussian-splatting'))

import torch, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera


class PipelineParamsDummy:
    convert_SHs_python = False; compute_cov3D_python = False; debug = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_views", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    # Load model
    gaussians = GaussianModel(3)
    gaussians.load_ply(os.path.join(args.model_dir, "point_cloud", "iteration_30000", "point_cloud.ply"))

    # Load scene info
    with open(os.path.join(args.model_dir, "cameras.json")) as f:
        cams = json.load(f)
    H, W = cams[0]["height"], cams[0]["width"]
    fx, fy = cams[0]["fx"], cams[0]["fy"]

    scale = 0.5
    H2, W2 = int(H*scale), int(W*scale)
    fov_x = 2*math.atan(W2/(2*fx*scale))
    fov_y = 2*math.atan(H2/(2*fy*scale))

    # Scene center from look-at points
    positions = np.array([c['position'] for c in cams])
    rots = np.array([c['rotation'] for c in cams])
    look_pts = positions - rots[:,2,:] * 1.0
    scene_center = np.median(look_pts, axis=0)
    base_dist = np.median(np.linalg.norm(positions - scene_center, axis=1))

    bg = torch.tensor([0.,0.,0.], dtype=torch.float32, device=device)
    N = args.num_views

    for i in tqdm(range(N), desc="Cinematic flight"):
        t = i / N  # 0 -> 1

        # Phase 1 (0-30%): approach from distance, slight spiral
        # Phase 2 (30-70%): circle around at medium distance
        # Phase 3 (70-100%): pull back and rise

        if t < 0.3:
            phase_t = t / 0.3
            angle = phase_t * math.pi * 1.5
            dist = base_dist * (2.0 - phase_t * 1.0)
            height = scene_center[1] - 0.3 + phase_t * 0.3
        elif t < 0.7:
            phase_t = (t - 0.3) / 0.4
            angle = math.pi * 1.5 + phase_t * math.pi * 2
            dist = base_dist * (1.0 + 0.1 * math.sin(phase_t * math.pi * 4))
            height = scene_center[1] + 0.1 * math.sin(phase_t * math.pi * 3)
        else:
            phase_t = (t - 0.7) / 0.3
            angle = math.pi * 1.5 + math.pi * 2 + phase_t * math.pi
            dist = base_dist * (1.0 + phase_t * 0.8)
            height = scene_center[1] + 0.1 + phase_t * 0.5

        cx = scene_center[0] + dist * math.cos(angle)
        cy = height
        cz = scene_center[2] + dist * math.sin(angle)
        pos = np.array([cx, cy, cz], dtype=np.float64)

        # Look at scene center
        look_at = scene_center + np.array([0, 0.05*math.sin(t*math.pi*8), 0], dtype=np.float64)
        forward = look_at - pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 1, 0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        R = np.stack([right, up, -forward], axis=0)
        T = -R @ pos

        dummy_img = Image.new('RGB', (W2, H2))
        cam = Camera(resolution=(H2,W2), colmap_id=0, R=R.astype(np.float64), T=T.astype(np.float64),
                    FoVx=fov_x, FoVy=fov_y, depth_params=None, image=dummy_img, invdepthmap=None,
                    image_name=f"c_{i:04d}", uid=i)

        with torch.no_grad():
            result = render(cam, gaussians, PipelineParamsDummy(), bg)
            frame = result["render"].clamp(0,1).permute(1,2,0).cpu().numpy()
        Image.fromarray((frame*255).astype(np.uint8)).save(os.path.join(args.out_dir, f"frame_{i:04d}.png"))

    print(f"\nDone. {N} frames -> {args.out_dir}")
    print(f"ffmpeg: ffmpeg -framerate 30 -i {args.out_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {args.out_dir}/cinematic.mp4")


if __name__ == "__main__":
    main()
