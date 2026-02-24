"""
FSA-MFD 测试/推理脚本

使用方法:
    # 在测试集上评估
    python test.py --config configs/levir_cd.yaml --checkpoint outputs/levir_cd/best_model.pth

    # 对单张图像对推理
    python test.py --img_t path/to/t1.png --img_t1 path/to/t2.png \
                   --checkpoint outputs/levir_cd/best_model.pth \
                   --output result.png
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from datasets import build_dataloader
from utils import (
    ChangeDetectionMetrics, setup_logger,
    save_change_map, visualize_comparison, set_random_seed
)


def parse_args():
    parser = argparse.ArgumentParser(description='FSA-MFD Testing/Inference')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU 设备号')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--save_pred', action='store_true',
                        help='保存预测图像')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='结果输出目录')
    # 单图推理参数
    parser.add_argument('--img_t', type=str, default=None,
                        help='时相1图像路径 (单图推理模式)')
    parser.add_argument('--img_t1', type=str, default=None,
                        help='时相2图像路径 (单图推理模式)')
    parser.add_argument('--output', type=str, default='./output.png',
                        help='单图推理输出路径')
    parser.add_argument('--img_size', type=int, default=256,
                        help='输入图像尺寸')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_image(img_path: str, img_size: int) -> torch.Tensor:
    """预处理单张图像"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean, std)
    return tensor.unsqueeze(0)  # [1, 3, H, W]


@torch.no_grad()
def test_on_dataset(model, loader, device, threshold, save_pred, output_dir, logger):
    """在数据集上测试并计算指标"""
    model.eval()
    metrics = ChangeDetectionMetrics(threshold=threshold)

    if save_pred:
        pred_dir = Path(output_dir) / 'predictions'
        vis_dir = Path(output_dir) / 'visualizations'
        pred_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    total_samples = 0

    for step, batch in enumerate(loader):
        img_t = batch['img_t'].to(device, non_blocking=True)
        img_t1 = batch['img_t1'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)
        names = batch['name']

        t0 = time.time()
        pred = model(img_t, img_t1)  # [B, 1, H, W]
        elapsed = time.time() - t0
        total_time += elapsed
        total_samples += img_t.size(0)

        metrics.update(pred, label)

        if save_pred:
            for i, name in enumerate(names):
                # 保存预测二值图
                save_change_map(
                    pred[i],
                    str(pred_dir / f'{name}.png'),
                    threshold=threshold
                )
                # 保存对比可视化
                visualize_comparison(
                    img_t[i], img_t1[i], pred[i], label[i],
                    str(vis_dir / f'{name}_vis.png')
                )

        if (step + 1) % 20 == 0:
            logger.info(f'Testing [{step+1}/{len(loader)}]')

    final_metrics = metrics.compute()
    avg_inference_ms = (total_time / total_samples) * 1000

    logger.info('\n' + '='*50)
    logger.info('Test Results:')
    logger.info(f'  F1:        {final_metrics["F1"]:.4f}%')
    logger.info(f'  IoU:       {final_metrics["IoU"]:.4f}%')
    logger.info(f'  OA:        {final_metrics["OA"]:.4f}%')
    logger.info(f'  Precision: {final_metrics["Precision"]:.4f}%')
    logger.info(f'  Recall:    {final_metrics["Recall"]:.4f}%')
    logger.info(f'  Avg Infer: {avg_inference_ms:.1f}ms/sample')
    logger.info('='*50)

    return final_metrics


@torch.no_grad()
def infer_single(model, img_t_path: str, img_t1_path: str,
                 img_size: int, device, threshold: float, output_path: str):
    """单图对推理"""
    model.eval()

    img_t = preprocess_image(img_t_path, img_size).to(device)
    img_t1 = preprocess_image(img_t1_path, img_size).to(device)

    pred = model(img_t, img_t1)  # [1, 1, H, W]
    pred_np = pred.squeeze().cpu().numpy()
    binary = (pred_np > threshold).astype(np.uint8) * 255

    Image.fromarray(binary).save(output_path)
    print(f'✓ Saved change map: {output_path}')
    print(f'  Change ratio: {binary.mean()/255*100:.2f}%')


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 单图推理模式 ----
    if args.img_t and args.img_t1:
        # 构建默认配置
        cfg_model = {
            'input_size': args.img_size,
            'pretrained': False,
        }
        model = build_model(cfg_model).to(device)

        # 加载权重
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'✓ Loaded checkpoint: {args.checkpoint}')

        infer_single(model, args.img_t, args.img_t1,
                     args.img_size, device, args.threshold, args.output)
        return

    # ---- 数据集测试模式 ----
    if args.config is None:
        raise ValueError('请提供 --config 参数 (数据集测试模式) 或 --img_t/--img_t1 (单图推理模式)')

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, name='test')
    logger.info(f'Config: {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')
    logger.info(f'Device: {device}')

    # 构建模型
    model = build_model(cfg['model']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info('✓ Model loaded successfully')

    # 构建测试 DataLoader
    ds_cfg = cfg['dataset']
    test_loader = build_dataloader(
        dataset_name=ds_cfg['name'],
        root=ds_cfg['root'],
        split='test',
        img_size=ds_cfg['img_size'],
        batch_size=1,  # 测试时 batch=1 以精确计时
        num_workers=4,
    )

    # 测试
    final_metrics = test_on_dataset(
        model, test_loader, device,
        threshold=args.threshold,
        save_pred=args.save_pred,
        output_dir=args.output_dir,
        logger=logger
    )

    # 保存结果
    import json
    result_file = Path(args.output_dir) / 'test_results.json'
    with open(result_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f'Results saved to: {result_file}')


if __name__ == '__main__':
    main()
