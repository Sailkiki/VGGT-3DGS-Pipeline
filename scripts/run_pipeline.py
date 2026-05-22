#!/usr/bin/env python3
"""
VGGT → 3DGS 完整管线运行脚本

用法:
  # Step 1: VGGT 推理 + 导出 COLMAP 格式
  python run_pipeline.py vggt --scene_dir ../data/processed --model_path ../models/model.pt

  # Step 2: 3DGS 训练 (使用 VGGT 初始化)
  python run_pipeline.py train_3dgs --data_dir ../data/processed --output_dir ../3dgs_output/vggt_init

  # Step 3: 3DGS 渲染评估
  python run_pipeline.py render --model_dir ../3dgs_output/vggt_init

  # 完整流程
  python run_pipeline.py all --scene_dir ../data/processed --model_path ../models/model.pt --output_dir ../3dgs_output/vggt_init
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VGGT_DIR = PROJECT_ROOT / "vggt"
GS_DIR = PROJECT_ROOT / "gaussian-splatting"


def run_cmd(cmd, desc, cwd=None):
    print(f"\n{'='*60}")
    print(f"[RUNNING] {desc}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[FAILED] {desc}")
        sys.exit(1)
    print(f"[DONE] {desc}")


def step_vggt(scene_dir, model_path, use_ba=False, seed=42):
    """
    运行 VGGT 推理，输出 COLMAP 格式到 <scene_dir>/sparse/
    """
    # 修改 demo_colmap.py 使其支持本地模型路径
    demo_path = VGGT_DIR / "demo_colmap.py"

    cmd = [
        sys.executable, str(demo_path),
        "--scene_dir", str(Path(scene_dir).resolve()),
        "--seed", str(seed),
    ]
    if use_ba:
        cmd.append("--use_ba")

    # 设置环境变量传递本地模型路径
    env = os.environ.copy()
    if model_path:
        env["VGGT_MODEL_PATH"] = str(Path(model_path).resolve())

    run_cmd(cmd, f"VGGT inference (scene={scene_dir})", env=env, cwd=str(VGGT_DIR))


def step_train_3dgs(data_dir, output_dir, iterations=30000):
    """
    使用 VGGT/COLMAP 的 sparse/ 结果训练 3DGS
    """
    train_script = GS_DIR / "train.py"
    cmd = [
        sys.executable, str(train_script),
        "-s", str(Path(data_dir).resolve()),
        "-m", str(Path(output_dir).resolve()),
        "--iterations", str(iterations),
    ]
    run_cmd(cmd, f"3DGS training (output={output_dir})", cwd=str(GS_DIR))


def step_render(model_dir, skip_train=True):
    """
    渲染测试视角
    """
    render_script = GS_DIR / "render.py"
    cmd = [
        sys.executable, str(render_script),
        "-m", str(Path(model_dir).resolve()),
    ]
    if skip_train:
        cmd.append("--skip_train")
    run_cmd(cmd, f"3DGS rendering (model={model_dir})", cwd=str(GS_DIR))


def step_colmap_baseline(data_dir, output_dir):
    """
    运行 COLMAP 作为基线
    """
    images_dir = Path(data_dir).resolve() / "images"
    sparse_dir = Path(output_dir).resolve() / "sparse"
    db_path = Path(output_dir).resolve() / "database.db"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # 特征提取
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--SiftExtraction.num_threads", "16",
    ], "COLMAP feature extraction")

    # 特征匹配
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.num_threads", "16",
    ], "COLMAP exhaustive matching")

    # 稀疏重建
    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ], "COLMAP sparse reconstruction")

    # 确保 3DGS 可读格式
    src_sparse = Path(sparse_dir) / "0"
    if not src_sparse.exists():
        print("[WARNING] COLMAP mapper did not produce sparse/0/ — reconstruction may have failed")


def step_evaluate(vggt_model_dir, colmap_model_dir, output_dir):
    """
    比较 VGGT 初始化和 COLMAP 初始化的结果
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_script = GS_DIR / "metrics.py"

    print("\n--- VGGT Init Results ---")
    vggt_results = Path(vggt_model_dir) / "test" / "results.json"
    if vggt_results.exists():
        import json
        with open(vggt_results) as f:
            data = json.load(f)
        for img_data in data:
            print(f"  {img_data['image_name']}: PSNR={img_data.get('PSNR','N/A'):.2f}, "
                  f"SSIM={img_data.get('SSIM','N/A'):.4f}, LPIPS={img_data.get('LPIPS','N/A'):.4f}")

    print("\n--- COLMAP Init Results ---")
    colmap_results = Path(colmap_model_dir) / "test" / "results.json"
    if colmap_results.exists():
        import json
        with open(colmap_results) as f:
            data = json.load(f)
        for img_data in data:
            print(f"  {img_data['image_name']}: PSNR={img_data.get('PSNR','N/A'):.2f}, "
                  f"SSIM={img_data.get('SSIM','N/A'):.4f}, LPIPS={img_data.get('LPIPS','N/A'):.4f}")


def main():
    parser = argparse.ArgumentParser(description="VGGT → 3DGS Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # VGGT step
    parser_vggt = subparsers.add_parser("vggt", help="Run VGGT inference + COLMAP export")
    parser_vggt.add_argument("--scene_dir", required=True, help="Directory containing images/ folder")
    parser_vggt.add_argument("--model_path", default=None, help="Path to local model.pt file")
    parser_vggt.add_argument("--use_ba", action="store_true", help="Use bundle adjustment")
    parser_vggt.add_argument("--seed", type=int, default=42)

    # 3DGS training step
    parser_train = subparsers.add_parser("train_3dgs", help="Train 3DGS")
    parser_train.add_argument("--data_dir", required=True, help="Data directory with sparse/0/")
    parser_train.add_argument("--output_dir", required=True, help="Output directory for model")
    parser_train.add_argument("--iterations", type=int, default=30000)

    # Render step
    parser_render = subparsers.add_parser("render", help="Render test views")
    parser_render.add_argument("--model_dir", required=True, help="3DGS model directory")
    parser_render.add_argument("--skip_train", action="store_true", default=True)

    # COLMAP baseline
    parser_colmap = subparsers.add_parser("colmap", help="Run COLMAP baseline")
    parser_colmap.add_argument("--data_dir", required=True, help="Data directory with images/")
    parser_colmap.add_argument("--output_dir", required=True)

    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Compare results")
    parser_eval.add_argument("--vggt_model_dir", required=True)
    parser_eval.add_argument("--colmap_model_dir", required=True)
    parser_eval.add_argument("--output_dir", default="./results")

    # Full pipeline
    parser_all = subparsers.add_parser("all", help="Run full pipeline")
    parser_all.add_argument("--scene_dir", required=True)
    parser_all.add_argument("--model_path", default=None)
    parser_all.add_argument("--output_dir", required=True)
    parser_all.add_argument("--iterations", type=int, default=30000)
    parser_all.add_argument("--use_ba", action="store_true")

    args = parser.parse_args()

    if args.command == "vggt":
        step_vggt(args.scene_dir, args.model_path, args.use_ba, args.seed)
    elif args.command == "train_3dgs":
        step_train_3dgs(args.data_dir, args.output_dir, args.iterations)
    elif args.command == "render":
        step_render(args.model_dir, args.skip_train)
    elif args.command == "colmap":
        step_colmap_baseline(args.data_dir, args.output_dir)
    elif args.command == "evaluate":
        step_evaluate(args.vggt_model_dir, args.colmap_model_dir, args.output_dir)
    elif args.command == "all":
        print("=== Phase 1/3: VGGT Inference ===")
        step_vggt(args.scene_dir, args.model_path, args.use_ba)
        print("\n=== Phase 2/3: 3DGS Training ===")
        step_train_3dgs(args.scene_dir, args.output_dir, args.iterations)
        print("\n=== Phase 3/3: 3DGS Rendering ===")
        step_render(args.output_dir)
        print(f"\nPipeline complete. Results in {args.output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
