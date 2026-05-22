#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    consistency_weight = get_expon_lr_func(opt.consistency_weight_init, opt.consistency_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_Lconsistency_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization (VGGT depth supervision)
        # Only compute every N iterations — depth changes slowly,
        # and the second render pass doubles per-iteration cost.
        need_depth_sup = depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and (iteration % 50 == 0)
        need_consistency = consistency_weight(iteration) > 0 and iteration >= opt.consistency_start_iter and (iteration % 100 == 0)

        Ll1depth_pure = 0.0
        Ll1depth = 0.0
        Lconsistency = 0.0

        if need_depth_sup or need_consistency:
            # --- Render depth from current viewpoint (shared) ---
            with torch.no_grad():
                means3D = gaussians.get_xyz.detach()
                w2v_cur = viewpoint_cam.world_view_transform
                ones = torch.ones(means3D.shape[0], 1, device=means3D.device)
                cam_xyz = torch.cat([means3D, ones], dim=-1) @ w2v_cur
                z_depth = cam_xyz[:, 2]
                z_min, z_max = z_depth.min(), z_depth.max()
                if z_max - z_min < 1e-6:
                    z_max = z_min + 1.0
                z_norm = (z_depth - z_min) / (z_max - z_min)
                depth_rgb = z_norm[:, None].repeat(1, 3).contiguous()

            depth_render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                       override_color=depth_rgb,
                                       use_trained_exp=False,
                                       separate_sh=False)
            rendered_depth_norm = depth_render_pkg["render"][0:1]  # [1, H, W]
            rendered_depth_cur = rendered_depth_norm * (z_max - z_min) + z_min

            # --- VGGT depth supervision ---
            if need_depth_sup:
                invDepth = 1.0 / (rendered_depth_cur + 1e-8)
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()

            # --- Multi-view consistency loss ---
            if need_consistency:
                nbr_cam = scene.neighbor_map[viewpoint_cam.uid]
                H, W = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)

                # Render depth from neighbor viewpoint
                with torch.no_grad():
                    w2v_nbr = nbr_cam.world_view_transform
                    cam_xyz_nbr = torch.cat([means3D, ones], dim=-1) @ w2v_nbr
                    z_depth_nbr = cam_xyz_nbr[:, 2]
                    z_min_nbr, z_max_nbr = z_depth_nbr.min(), z_depth_nbr.max()
                    if z_max_nbr - z_min_nbr < 1e-6:
                        z_max_nbr = z_min_nbr + 1.0
                    z_norm_nbr = (z_depth_nbr - z_min_nbr) / (z_max_nbr - z_min_nbr)
                    depth_rgb_nbr = z_norm_nbr[:, None].repeat(1, 3).contiguous()

                depth_render_pkg_nbr = render(nbr_cam, gaussians, pipe, background,
                                               override_color=depth_rgb_nbr,
                                               use_trained_exp=False,
                                               separate_sh=False)
                rendered_depth_nbr = depth_render_pkg_nbr["render"][0:1] * (z_max_nbr - z_min_nbr) + z_min_nbr

                # Build camera intrinsics from FoV
                from utils.graphics_utils import fov2focal
                fx_cur = fov2focal(viewpoint_cam.FoVx, W)
                fy_cur = fov2focal(viewpoint_cam.FoVy, H)
                fx_nbr = fov2focal(nbr_cam.FoVx, W)
                fy_nbr = fov2focal(nbr_cam.FoVy, H)

                K_cur = torch.tensor([[fx_cur, 0, W / 2], [0, fy_cur, H / 2], [0, 0, 1]],
                                      device="cuda", dtype=torch.float32)
                K_nbr = torch.tensor([[fx_nbr, 0, W / 2], [0, fy_nbr, H / 2], [0, 0, 1]],
                                      device="cuda", dtype=torch.float32)

                W2V_cur = viewpoint_cam.world_view_transform
                W2V_nbr = nbr_cam.world_view_transform
                V2W_cur = torch.inverse(W2V_cur)

                # Pixel grid for current view
                y, x = torch.meshgrid(
                    torch.arange(H, device="cuda", dtype=torch.float32),
                    torch.arange(W, device="cuda", dtype=torch.float32), indexing='ij')
                pixels_cur = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]

                # Backproject: cam_pt = D * K^-1 @ pixel
                invK_cur = torch.inverse(K_cur)
                cam_rays = (invK_cur @ pixels_cur.reshape(-1, 3).T).T.reshape(H, W, 3)
                cam_pts = cam_rays * rendered_depth_cur.squeeze().unsqueeze(-1)  # [H, W, 3]

                # Transform to world
                cam_pts_h = torch.cat([cam_pts, torch.ones(H, W, 1, device="cuda")], dim=-1)
                world_pts = (V2W_cur @ cam_pts_h.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]

                # Project to neighbor camera
                world_pts_h = torch.cat([world_pts, torch.ones(H, W, 1, device="cuda")], dim=-1)
                nbr_cam_pts = (W2V_nbr @ world_pts_h.reshape(-1, 4).T).T.reshape(H, W, 4)[..., :3]
                expected_depth = nbr_cam_pts[..., 2]  # [H, W]

                # Project to neighbor pixel
                nbr_pixels_h = (K_nbr @ nbr_cam_pts.reshape(-1, 3).T).T.reshape(H, W, 3)
                nbr_pixels = nbr_pixels_h[..., :2] / (nbr_pixels_h[..., 2:3] + 1e-8)

                # Sample neighbor depth at projected locations
                nbr_grid = torch.stack([
                    nbr_pixels[..., 0] / (W - 1) * 2 - 1,
                    nbr_pixels[..., 1] / (H - 1) * 2 - 1
                ], dim=-1).unsqueeze(0)  # [1, H, W, 2]

                D_nbr = rendered_depth_nbr.squeeze().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                D_nbr_sampled = F.grid_sample(D_nbr, nbr_grid, mode='bilinear',
                                               padding_mode='zeros', align_corners=True).squeeze()

                # Valid mask
                valid = (nbr_pixels[..., 0] >= 0) & (nbr_pixels[..., 0] < W) & \
                        (nbr_pixels[..., 1] >= 0) & (nbr_pixels[..., 1] < H) & \
                        (expected_depth > 0.01) & (D_nbr_sampled > 0.01)

                if valid.sum() > 100:
                    diff = torch.abs(D_nbr_sampled - expected_depth)
                    rel_diff = diff / (expected_depth + 1e-8)
                    valid = valid & (rel_diff < 0.1)  # mask occlusions

                    if valid.sum() > 100:
                        Lconsistency_pure = diff[valid].mean()
                        Lconsistency = consistency_weight(iteration) * Lconsistency_pure
                        loss += Lconsistency
                        Lconsistency = Lconsistency.item()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_Lconsistency_for_log = 0.4 * Lconsistency + 0.6 * ema_Lconsistency_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Consist": f"{ema_Lconsistency_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # Track per-Gaussian max contribution across views
                gaussians.accumulate_contribution(visibility_filter, radii)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Contribution-based redundancy pruning
            if iteration == opt.contribution_prune_iter:
                n_pruned = gaussians.prune_by_contribution(opt.contribution_prune_ratio)
                print(f"\n[ITER {iteration}] Contribution prune: removed {n_pruned} Gaussians ({opt.contribution_prune_ratio*100:.0f}%)")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
