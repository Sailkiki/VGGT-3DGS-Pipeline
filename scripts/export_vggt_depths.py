#!/usr/bin/env python3
"""
Export VGGT depth & confidence maps for 3DGS depth supervision.

Output:
  <scene>/depths/<name>.npy    — float32 inverse depth (H, W), loaded by camera_utils
  <scene>/depths/<name>.png    — uint16 viz copy (optional, for inspection)
  <scene>/sparse/0/depth_params.json — scale=inv_max, offset=0

Usage:
  conda activate vggt3d
  python scripts/export_vggt_depths.py --scene_dir data/processed_scene
"""

import argparse, os, sys, glob, json
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VGGT_DIR = PROJECT_ROOT / "vggt"
sys.path.insert(0, str(VGGT_DIR))

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def export_one_scene(scene_dir, model_path, seed=42):
    scene_dir = Path(scene_dir).resolve()
    image_dir = scene_dir / "images"
    depths_dir = scene_dir / "depths"
    sparse_dir = scene_dir / "sparse" / "0"

    if not image_dir.exists():
        print(f"  [SKIP] no images/ in {scene_dir}")
        return False

    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # ---- Load VGGT ----
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = VGGT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"  VGGT loaded")

    # ---- Load images ----
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    S = len(img_paths)
    if S == 0:
        print(f"  [SKIP] no images found")
        return False

    # VGGT always runs at 518
    vggt_res = 518
    images, original_coords = load_and_preprocess_images_square(img_paths, vggt_res)
    images = images.to(device)
    print(f"  Loaded {S} images (VGGT resolution: {vggt_res})")

    # ---- VGGT inference ----
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            agg, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(agg)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(agg, images[None], ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).squeeze(-1).cpu().numpy()      # [S, 518, 518]
    depth_conf = depth_conf.squeeze(0).cpu().numpy()                 # [S, 518, 518]
    original_coords_np = original_coords.cpu().numpy()
    img_names = [os.path.basename(p) for p in img_paths]

    depth_params = {}
    saved = 0

    for i in range(S):
        name_no_ext = os.path.splitext(img_names[i])[0]

        # original_coords format: [top, left, h_in_518, w_in_518, full_w, full_h]
        top = int(original_coords_np[i, 0])
        left = int(original_coords_np[i, 1])
        h_518 = int(original_coords_np[i, 2])   # image region height in 518 canvas
        w_518 = int(original_coords_np[i, 3])   # image region width in 518 canvas
        full_w = int(original_coords_np[i, 4])
        full_h = int(original_coords_np[i, 5])

        # Crop image region from padded square canvas
        vggt_depth_518 = depth_map[i]  # [518, 518]
        depth_cropped = vggt_depth_518[top:top+h_518, left:left+w_518]  # [h_518, w_518]

        # Resize to original full resolution
        d_t = torch.from_numpy(depth_cropped).float().unsqueeze(0).unsqueeze(0)
        depth_full = F.interpolate(d_t, size=(full_h, full_w), mode='bilinear',
                                   align_corners=False).squeeze().numpy()

        # Inverse depth
        inv_depth = 1.0 / np.maximum(depth_full, 1e-5)

        # Per-image max for reliability check in depth_params.json
        inv_max = float(inv_depth.max())
        # scale=1.0: keep VGGT inverse depth in original 1/m units (matches rendered invDepth)
        depth_params[name_no_ext] = {"scale": 1.0, "offset": 0.0, "inv_max": inv_max if inv_max > 0 else 1.0}

        # Save float32 .npy (fast, precise — primary format)
        np.save(os.path.join(depths_dir, f"{name_no_ext}.npy"), inv_depth.astype(np.float32))

        # ---- Export confidence map (soft weight for depth loss) ----
        vggt_conf_518 = depth_conf[i]  # [518, 518]
        conf_cropped = vggt_conf_518[top:top+h_518, left:left+w_518]
        c_t = torch.from_numpy(conf_cropped).float().unsqueeze(0).unsqueeze(0)
        conf_full = F.interpolate(c_t, size=(full_h, full_w), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        # Normalize to [0.05, 1.0] per-image using 5th/95th percentiles
        c_low, c_high = np.percentile(conf_full, [10, 95])
        if c_high > c_low + 1e-6:
            conf_norm = np.clip((conf_full - c_low) / (c_high - c_low), 0.0, 1.0)
        else:
            conf_norm = np.ones_like(conf_full) * 0.5
        conf_out = np.maximum(conf_norm, 0.05)  # floor at 0.05, never zero
        np.save(os.path.join(depths_dir, f"{name_no_ext}_conf.npy"), conf_out.astype(np.float32))

        saved += 1

    # ---- Write depth_params.json ----
    # Compute global median inv_max for reliability check in cameras.py
    inv_maxes = [v["inv_max"] for v in depth_params.values() if v.get("inv_max", 0) > 0]
    med_inv_max = float(np.median(inv_maxes)) if inv_maxes else 1.0
    for k in depth_params:
        depth_params[k]["med_inv_max"] = med_inv_max

    with open(os.path.join(sparse_dir, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)

    print(f"  Saved {saved} depth maps → {depths_dir}")
    print(f"  med_inv_max = {med_inv_max:.3f}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, action="append", default=[])
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "vggt" / "pretrained" / "model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.scene_dir:
        args.scene_dir = [
            str(PROJECT_ROOT / "data" / "processed_scene"),
            str(PROJECT_ROOT / "data" / "scene2_cuda"),
            str(PROJECT_ROOT / "data" / "video_scene"),
            str(PROJECT_ROOT / "data" / "video2_scene"),
        ]

    for sd in args.scene_dir:
        print(f"\n{'='*50}\nExporting: {sd}\n{'='*50}")
        export_one_scene(sd, args.model_path, args.seed)


if __name__ == "__main__":
    main()
