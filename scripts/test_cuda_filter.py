#!/usr/bin/env python3
"""
Tests for CUDA point cloud filter V2.
Tests are designed to detect specific code regressions via behavioral signals.
"""

import os
import sys
import numpy as np
import torch

os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


def _compute_center(ext):
    """Camera center from extrinsics [3,4]."""
    R = ext[:3, :3]
    t = ext[:3, 3]
    return -R.T @ t


def _call_consistency(points_3d, depth_maps, depth_conf, extrinsics, intrinsics,
                      neighbor_indices, cam_centers, consistency_thresh,
                      gradient_maps=None, lambda_grad=0.0):
    """Wrapper: handles the new kernel signature so tests don't all need updating."""
    import point_filter_v2
    S, H, W = points_3d.shape[:3]
    device = points_3d.device
    if gradient_maps is None:
        gradient_maps = torch.zeros(S, H, W, device=device)
    return point_filter_v2._point_filter_v2_ops.multiview_consistency_v2(
        points_3d.contiguous(),
        depth_maps.contiguous(),
        depth_conf.contiguous(),
        extrinsics.contiguous(),
        intrinsics.contiguous(),
        neighbor_indices,
        cam_centers,
        gradient_maps.contiguous(),
        consistency_thresh,
        lambda_grad,
    )


def test_bilinear_depth_interpolation():
    """
    Bug 1: V2 uses nearest-neighbor for depth sampling.

    A 3D point from frame 0 projects to a SUB-PIXEL location (u=33.4) in frame 1.
    The 4 surrounding depth pixels are set so that:
      - bilinear avg = 2.0 (exactly matches triangulated depth)
      - nearest-neighbor pixel = 1.0 (50% error, fails 5% threshold)

    Bilinear → score ≈ 1.0.  Nearest-neighbor → score ≈ 0.0.
    """
    print("\n=== TEST 1: Bilinear Depth Interpolation ===")

    S, H, W = 2, 64, 64
    device = torch.device("cuda")

    points_3d = torch.zeros(S, H, W, 3, device=device)
    depth_maps = torch.full((S, H, W), 2.0, device=device)
    depth_conf = torch.ones(S, H, W, device=device)

    # Intrinsics: fx=fy=32, cx=cy=32
    K = torch.tensor([[32., 0, 32.],
                       [0, 32., 32.],
                       [0,  0,  1.]], device=device)
    intrinsics = K.unsqueeze(0).repeat(S, 1, 1)

    # Extrinsics world→cam: frame0=identity, frame1=shifted 0.1m +X
    E0 = torch.tensor([[1., 0, 0, 0],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    E1 = torch.tensor([[1., 0, 0, -0.1],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    extrinsics = torch.stack([E0, E1])

    # Test pixel: frame 0, (h=32, w=35), depth=2.0
    # world_X = (35-32)*2/32 = 0.1875
    # In frame1 cam: X = 0.1875 - 0.1 = 0.0875
    # u = 32*0.0875/2.0 + 32 = 33.4  (fractional!)
    test_h, test_w = 32, 35
    px = (test_w - 32.0) * 2.0 / 32.0   # 0.1875
    py = (test_h - 32.0) * 2.0 / 32.0   # 0.0
    pz = 2.0

    points_3d[0, test_h, test_w, 0] = px
    points_3d[0, test_h, test_w, 1] = py
    points_3d[0, test_h, test_w, 2] = pz

    # u=33.4, v=32.0 in frame1. Bilinear corners: (33,32),(34,32),(33,33),(34,33)
    # wu=0.4, wv=0.0 → w00=0.6, w10=0.4
    # Set: depth[32,33]=1.0, depth[32,34]=3.5
    # bilinear: 0.6*1.0 + 0.4*3.5 = 2.0 ✓
    # nn(33.4)=33: depth=1.0 → error=50% >> 5%
    depth_maps[1, 32, 33] = 1.0
    depth_maps[1, 32, 34] = 3.5

    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int32, device=device)

    c0 = _compute_center(E0.cpu().numpy())
    c1 = _compute_center(E1.cpu().numpy())
    cam_centers = torch.tensor(np.stack([c0, c1]), dtype=torch.float32, device=device)

    consistency_score = _call_consistency(
        points_3d, depth_maps, depth_conf, extrinsics, intrinsics,
        neighbor_indices, cam_centers, 0.05,
    )

    test_idx = 0 * H * W + test_h * W + test_w
    score = consistency_score[test_idx].item()
    print(f"  Consistency score at test pixel: {score:.4f}")

    if score > 0.9:
        print("  PASS: Bilinear interpolation detected (score ≈ 1.0)")
        return True
    elif score < 0.1:
        print("  FAIL: Nearest-neighbor detected! (score ≈ 0, expected ≈ 1.0)")
        print("  → Bug 1 CONFIRMED: V2 kernel uses nearest-neighbor for depth sampling")
        return False
    else:
        print(f"  INCONCLUSIVE: score={score:.4f}")
        return False


def test_pure_geometric_score():
    """
    Bug 2: Kernel multiplies geometric consistency by depth_conf.

    A point with perfect geometry across all neighbors but LOW depth_conf
    should still get a high score (pure geometric = 1.0).
    Currently the kernel does `frac * conf`, so low conf kills the score.

    Test: set conf=0.3, make geometry perfect → score should be 0.0 (bug) or 1.0 (fix).
    No — actually with bilinear fixed, we want to isolate Bug 2. We make a point
    that is geometrically perfect (all neighbor depths agree). Set conf to 0.2.
    Bug predicts: score = 1.0 * 0.2 = 0.2.  Fix predicts: score = 1.0.
    """
    print("\n=== TEST 2: Pure Geometric Score (no conf multiplication) ===")

    S, H, W = 2, 64, 64
    device = torch.device("cuda")

    points_3d = torch.zeros(S, H, W, 3, device=device)
    depth_maps = torch.full((S, H, W), 2.0, device=device)
    depth_conf = torch.ones(S, H, W, device=device)

    K = torch.tensor([[32., 0, 32.],
                       [0, 32., 32.],
                       [0,  0,  1.]], device=device)
    intrinsics = K.unsqueeze(0).repeat(S, 1, 1)

    E0 = torch.tensor([[1., 0, 0, 0],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    E1 = torch.tensor([[1., 0, 0, -0.1],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    extrinsics = torch.stack([E0, E1])

    # Test pixel: frame 0 (32, 35), depth=2.0
    # Projects to frame 1 at u=33.4, v=32.0 (sub-pixel, covered by Bug 1 fix)
    test_h, test_w = 32, 35
    points_3d[0, test_h, test_w, 0] = (test_w - 32.0) * 2.0 / 32.0
    points_3d[0, test_h, test_w, 1] = (test_h - 32.0) * 2.0 / 32.0
    points_3d[0, test_h, test_w, 2] = 2.0

    # ALL depth values = 2.0 → bilinear avg = 2.0 → geometric error = 0%
    # No need for special values anymore since bilinear is fixed
    depth_maps[1, 32, 33] = 2.0
    depth_maps[1, 32, 34] = 2.0

    # Set LOW confidence at test pixel
    LOW_CONF = 0.2
    depth_conf[0, test_h, test_w] = LOW_CONF

    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int32, device=device)

    c0 = _compute_center(E0.cpu().numpy())
    c1 = _compute_center(E1.cpu().numpy())
    cam_centers = torch.tensor(np.stack([c0, c1]), dtype=torch.float32, device=device)

    consistency_score = _call_consistency(
        points_3d, depth_maps, depth_conf, extrinsics, intrinsics,
        neighbor_indices, cam_centers, 0.05,
    )

    test_idx = 0 * H * W + test_h * W + test_w
    score = consistency_score[test_idx].item()
    print(f"  Consistency score at test pixel (conf={LOW_CONF}): {score:.4f}")

    # Bug: score = frac * conf = 1.0 * 0.2 = 0.2
    # Fix: score = frac = 1.0
    if abs(score - 1.0) < 0.01:
        print("  PASS: Pure geometric score (not contaminated by confidence)")
        return True
    elif abs(score - LOW_CONF) < 0.05:
        print(f"  FAIL: Score = conf = {LOW_CONF}! Kernel is multiplying by confidence")
        print("  → Bug 2 CONFIRMED: consistency_score = frac * conf instead of frac")
        return False
    else:
        print(f"  INCONCLUSIVE: score={score:.4f}, expected 1.0 (fix) or {LOW_CONF} (bug)")
        return False


def test_bilinear_color_interpolation():
    """
    Bug 3: V2 uses nearest-neighbor for color sampling in fusion kernel.

    A 3D point from frame 0 projects to sub-pixel (u=33.4, v=32.0) in frame 1.
    4 surrounding pixels: (33,32)=Red, (34,32)=Blue. Source pixel = Black.
    Bilinear neighbor = 0.6*Red + 0.4*Blue = [153,0,102].
    Nearest-neighbor (u=33) = Red = [255,0,0].

    Source weight=1.0, neighbor angle_weight≈0.9987.
    Final ≈ neighbor * 0.9987/(1+0.9987) ≈ neighbor * 0.5.
    Bilinear: ≈ [76.5, 0, 51].  NN: ≈ [127.4, 0, 0].
    """
    print("\n=== TEST 3: Bilinear Color Interpolation ===")

    S, H, W = 2, 64, 64
    device = torch.device("cuda")

    points_3d = torch.zeros(S, H, W, 3, device=device)
    depth_maps = torch.full((S, H, W), 2.0, device=device)
    depth_conf = torch.ones(S, H, W, device=device)
    images = torch.zeros(S, H, W, 3, device=device)

    K = torch.tensor([[32., 0, 32.],
                       [0, 32., 32.],
                       [0,  0,  1.]], device=device)
    intrinsics = K.unsqueeze(0).repeat(S, 1, 1)

    E0 = torch.tensor([[1., 0, 0, 0],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    E1 = torch.tensor([[1., 0, 0, -0.1],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    extrinsics = torch.stack([E0, E1])

    test_h, test_w = 32, 35
    points_3d[0, test_h, test_w, 0] = (test_w - 32.0) * 2.0 / 32.0  # 0.1875
    points_3d[0, test_h, test_w, 1] = (test_h - 32.0) * 2.0 / 32.0  # 0.0
    points_3d[0, test_h, test_w, 2] = 2.0

    # Source pixel = black (contributes zero visually)
    images[0, test_h, test_w] = torch.tensor([0., 0., 0.], device=device)

    # Frame 1: surrounding pixels at projected location (33.4, 32.0)
    # (33,32)=Red [255,0,0], (34,32)=Blue [0,0,255]
    images[1, 32, 33] = torch.tensor([255., 0., 0.], device=device)
    images[1, 32, 34] = torch.tensor([0., 0., 255.], device=device)

    consistency_score = torch.zeros(S * H * W, device=device)
    test_idx = 0 * H * W + test_h * W + test_w
    consistency_score[test_idx] = 1.0

    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int32, device=device)

    c0 = _compute_center(E0.cpu().numpy())
    c1 = _compute_center(E1.cpu().numpy())
    cam_centers = torch.tensor(np.stack([c0, c1]), dtype=torch.float32, device=device)

    import point_filter_v2
    fused_colors = point_filter_v2._point_filter_v2_ops.weighted_color_fusion_v2(
        points_3d.contiguous(),
        images.contiguous(),
        extrinsics.contiguous(),
        intrinsics.contiguous(),
        cam_centers,
        neighbor_indices,
        consistency_score.contiguous(),
        0.5,
    )

    color = fused_colors[test_idx].cpu().numpy()
    r, g, b = color[0], color[1], color[2]
    print(f"  Fused color: R={r:.1f} G={g:.1f} B={b:.1f}")

    # Bilinear: R ~76.5, B ~51.0. Nearest-neighbor: R ~127.4, B ~0.
    # Use midpoint as threshold: R < 100 → bilinear, R > 100 → NN
    if r < 100.0 and b > 20.0:
        print("  PASS: Bilinear color interpolation detected")
        return True
    elif r > 100.0 and b < 10.0:
        print("  FAIL: Nearest-neighbor color detected! (pure red, expected [~77,0,~51])")
        print("  → Bug 3 CONFIRMED: V2 kernel uses nearest-neighbor for color sampling")
        return False
    else:
        print(f"  INCONCLUSIVE: unexpected color [{r:.1f}, {g:.1f}, {b:.1f}]")
        return False


def test_full_pipeline():
    """
    Integration test: full PointCloudFilterV2.filter() pipeline.

    Creates 4-frame synthetic scene, runs the complete filter, checks:
    - No crashes
    - Scores in [0, 1]
    - Output point count reasonable
    - Output has correct dict keys
    """
    print("\n=== INTEGRATION TEST: Full Pipeline ===")

    S, H, W = 4, 64, 64
    device = torch.device("cuda")

    # Synthesize data: all cameras at origin looking forward, depth=2.0 everywhere
    points_3d = torch.zeros(S, H, W, 3, device=device)
    depth_maps = torch.full((S, H, W), 2.0, device=device)
    depth_conf = torch.ones(S, H, W, device=device)
    images = torch.zeros(S, H, W, 3, device=device)

    K = torch.tensor([[32., 0, 32.],
                       [0, 32., 32.],
                       [0,  0,  1.]], device=device)
    intrinsics = K.unsqueeze(0).repeat(S, 1, 1)

    # 4 frames, slight shifts
    extrinsics = torch.zeros(S, 3, 4, device=device)
    for i in range(S):
        E = torch.tensor([[1., 0, 0, float(i) * 0.05],
                           [0, 1., 0, 0.],
                           [0, 0, 1., 0.]], device=device)
        extrinsics[i] = E

    # Unproject depth to 3D points
    for s in range(S):
        for h in range(H):
            for w in range(W):
                x = (w - 32.0) * 2.0 / 32.0
                y = (h - 32.0) * 2.0 / 32.0
                z = 2.0
                points_3d[s, h, w, 0] = x
                points_3d[s, h, w, 1] = y
                points_3d[s, h, w, 2] = z

    # Fill images with gradient for testing color fusion
    for s in range(S):
        for h in range(H):
            for w in range(W):
                images[s, h, w, 0] = (float(w) / W) * 255.0
                images[s, h, w, 1] = (float(h) / H) * 255.0
                images[s, h, w, 2] = 128.0

    import point_filter_v2
    flt = point_filter_v2.PointCloudFilterV2(
        num_neighbors=2,
        consistency_thresh=0.05,
        target_points=1000,
        consistency_min=0.5,
    )

    result = flt.filter(
        points_3d=points_3d,
        depth_maps=depth_maps,
        depth_conf=depth_conf,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        images=images,
    )

    # Checks
    errors = []

    # 1. Required keys
    for key in ["points_3d", "points_rgb", "points_conf", "num_filtered", "num_selected"]:
        if key not in result:
            errors.append(f"Missing key: {key}")

    # 2. Score range (in [0, 1+lambda_grad] now that gradient bonus is added)
    conf = result.get("points_conf", None)
    if conf is not None:
        if conf.max() > 1.5 + 1e-5:
            errors.append(f"Max confidence > 1.5: {conf.max():.4f}")
        if conf.min() < 0.0 - 1e-5:
            errors.append(f"Min confidence < 0.0: {conf.min():.4f}")

    # 3. Point count
    n_selected = result.get("num_selected", 0)
    if n_selected == 0:
        errors.append("No points selected!")
    elif n_selected > 1000:
        errors.append(f"Too many points: {n_selected} (max 1000)")

    # 4. Points shape
    pts = result.get("points_3d", None)
    if pts is not None:
        if pts.shape[0] != n_selected:
            errors.append(f"Point count mismatch: {pts.shape[0]} vs {n_selected}")
        if pts.shape[1] != 3:
            errors.append(f"Points should be (N,3), got {pts.shape}")

    # 5. Colors shape
    rgb = result.get("points_rgb", None)
    if rgb is not None and rgb.shape != pts.shape:
        errors.append(f"Color shape mismatch: {rgb.shape} vs {pts.shape}")

    print(f"  Selected: {n_selected} points")
    if conf is not None:
        print(f"  Score range: [{conf.min():.4f}, {conf.max():.4f}]")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    else:
        print("  PASS: Full pipeline integration test")
        return True


def test_partial_bilinear():
    """
    Change 1: Partial bilinear allows edge points when not all 4 corners valid.

    Scenario: one corner depth=0 (edge case). Old code skips neighbor entirely.
    New code uses 3 valid corners. Score should be ~1.0 (depth still matches).
    """
    print("\n=== TEST 4: Partial Bilinear (edge resilience) ===")

    S, H, W = 2, 64, 64
    device = torch.device("cuda")

    points_3d = torch.zeros(S, H, W, 3, device=device)
    depth_maps = torch.full((S, H, W), 2.0, device=device)
    depth_conf = torch.ones(S, H, W, device=device)

    K = torch.tensor([[32., 0, 32.],
                       [0, 32., 32.],
                       [0,  0,  1.]], device=device)
    intrinsics = K.unsqueeze(0).repeat(S, 1, 1)

    E0 = torch.tensor([[1., 0, 0, 0],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    E1 = torch.tensor([[1., 0, 0, -0.1],
                        [0, 1., 0, 0],
                        [0, 0, 1., 0]], device=device)
    extrinsics = torch.stack([E0, E1])

    test_h, test_w = 32, 35
    points_3d[0, test_h, test_w, 0] = (test_w - 32.0) * 2.0 / 32.0
    points_3d[0, test_h, test_w, 1] = (test_h - 32.0) * 2.0 / 32.0
    points_3d[0, test_h, test_w, 2] = 2.0

    # u=33.4, v=32.0. Corners: (33,32),(34,32),(33,33),(34,33)
    # Weights: w00=0.6, w10=0.4, w01=0, w11=0
    # Set one corner to 0 (depth edge):
    # (33,32)=0.0 (INVALID!), (34,32)=2.0
    # valid_w = 0 + 0.4 = 0.4 — wait, that's < 0.5!
    # Let me use a different sub-pixel: u=33.25, v=32.0
    # Then wu=0.25, wv=0.0
    # w00=0.75, w10=0.25
    # Set (33,32)=0 (invalid): valid_w = 0 + 0.25 = 0.25 < 0.5 → STILL skip

    # Hmm, I need valid_w >= 0.5 with one invalid corner.
    # Use u=33.5, v=32.0. wu=0.5, w00=0.5, w10=0.5.
    # actually: wu=0.5, wv=0, w00=0.5, w10=0.5
    # If (33,32)=0: valid_w = 0.5 ≥ 0.5 → OK

    # Need u=33.5 → cam_x = (33.5-32)*2/32 = 0.09375
    # px_world = 0.09375 + 0.1 = 0.19375
    # test_w = 32 + 0.19375*32/2 = 35.1 → not integer
    # Let me use: px_world=0.1875 (test_w=35), cam_x=0.0875, u=33.4
    # wu=0.4, w00=0.6, w10=0.4
    # Set (33,32)=0: valid_w = 0 + 0.4 = 0.4 < 0.5 → skip

    # Use more equal weights. u=33.5. Need test_w=35.2... not integer.
    # OK, use nearest integer and accept approximate:
    # test_w=35 gives px_world=0.1875, u=33.4. Not ideal.
    # Let me instead use a case where 3 corners cover most weight.
    # u=33.6, v=32.3. wu=0.6, wv=0.3.
    # w00=0.4*0.7=0.28, w10=0.6*0.7=0.42, w01=0.4*0.3=0.12, w11=0.6*0.3=0.18
    # Set d00=0: valid_w = 0.42+0.12+0.18 = 0.72 ≥ 0.5 ✓

    # u=33.6 → cam_x_needed = (33.6-32)*2/32 = 0.1
    # px_world = 0.1 + 0.1 = 0.2
    # test_w = 32 + 0.2*32/2 = 35.2... close to 35
    # Let me just adjust pixel and camera shift to hit exact u=33.6.
    # Use camera shift 0.08: u = 32*(0.1875-0.08)/2 + 32 = 32*0.1075/2+32 = 33.72
    # Hmm.

    # Actually, let me use a simpler setup: make the camera shift such that
    # u comes out to an exact fractional value.

    # Let me just use u=33.5 (wu=0.5) with a suitable test_w.
    # u=33.5 → cam_x_needed = (33.5-32)*2/32 = 0.09375
    # With camera at 0.08: px_world = cam_x + 0.08 = 0.17375
    # test_w = 32 + 0.17375*32/2 = 34.78 → use 35 (close enough, u≈33.54)

    # Let me just test with approximate values and check the score.
    # Set d01 and d11 to something invalid, d00 and d10 valid.
    # Actually, let me simplify: use v=32.0 exactly:
    # wv=0, w00=1-wu, w10=wu, w01=w11=0 (no contribution)
    # Set d10=0: valid_w = w00 (if d00 valid) = 1-wu
    # Need 1-wu ≥ 0.5 → wu ≤ 0.5. OK, with u=33.4, wu=0.4, valid_w=0.6 ≥ 0.5 ✓

    # Let me use test_w=35 (u=33.4), set d10 (=depth at (34,32)) = 0
    # d00=depth[32,33]=2.0, d10=depth[32,34]=0.0 (INVALID)
    # valid_w = 0.6, depth_sum = 0.6*2.0 = 1.2
    # vggt_depth = 1.2/0.6 = 2.0 ✓ (matches cam_z)

    depth_maps[1, 32, 33] = 2.0  # d00 — valid
    depth_maps[1, 32, 34] = 0.0  # d10 — INVALID (depth edge!)

    neighbor_indices = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
    c0 = _compute_center(E0.cpu().numpy())
    c1 = _compute_center(E1.cpu().numpy())
    cam_centers = torch.tensor(np.stack([c0, c1]), dtype=torch.float32, device=device)

    consistency_score = _call_consistency(
        points_3d, depth_maps, depth_conf, extrinsics, intrinsics,
        neighbor_indices, cam_centers, 0.05,
    )

    test_idx = 0 * H * W + test_h * W + test_w
    score = consistency_score[test_idx].item()
    print(f"  Consistency score (one corner invalid): {score:.4f}")

    if score > 0.9:
        print("  PASS: Partial bilinear works — edge point not skipped")
        return True
    elif score < 0.1:
        print("  FAIL: Partial bilinear NOT working — edge point still skipped")
        return False
    else:
        print(f"  INCONCLUSIVE: score={score:.4f}")
        return False


if __name__ == "__main__":
    print("CUDA Filter V2 — Regression Tests")
    print("=" * 60)
    results = {}
    results["bilinear_depth"] = test_bilinear_depth_interpolation()
    results["pure_geometric"] = test_pure_geometric_score()
    results["bilinear_color"] = test_bilinear_color_interpolation()
    results["partial_bilinear"] = test_partial_bilinear()
    results["full_pipeline"] = test_full_pipeline()
    print("\n" + "=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
