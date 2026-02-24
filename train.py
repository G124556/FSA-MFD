"""
FSA-MFD 训练脚本

使用方法:
    python train.py --config configs/levir_cd.yaml
    python train.py --config configs/whu_cd.yaml --resume outputs/whu_cd/latest.pth
    python train.py --config configs/cdd.yaml --gpu 0,1
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from datasets import build_dataloader
from utils import (
    ChangeDetectionMetrics, AverageMeter,
    setup_logger, CheckpointManager,
    count_parameters, set_random_seed, EarlyStopping
)


def parse_args():
    parser = argparse.ArgumentParser(description='FSA-MFD Training')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (YAML)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU 设备号, 多卡用逗号分隔, e.g., "0,1"')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (覆盖配置文件)')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式 (仅运行少量批次)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_one_epoch(model, loader, optimizer, scaler, device,
                    epoch, logger, use_amp=True, debug=False):
    """训练一个 epoch"""
    model.train()

    loss_meter = AverageMeter('Loss')
    ce_meter = AverageMeter('CE')
    cl_meter = AverageMeter('CL')

    start_time = time.time()

    for step, batch in enumerate(loader):
        if debug and step >= 5:
            break

        img_t = batch['img_t'].to(device, non_blocking=True)
        img_t1 = batch['img_t1'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            _, total_loss, ce_loss, cl_loss = model(img_t, img_t1, label)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = img_t.size(0)
        loss_meter.update(total_loss.item(), bs)
        ce_meter.update(ce_loss.item(), bs)
        cl_meter.update(cl_loss.item(), bs)

        if (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f'Epoch [{epoch}] Step [{step+1}/{len(loader)}] '
                f'Loss={loss_meter.avg:.4f} '
                f'CE={ce_meter.avg:.4f} '
                f'CL={cl_meter.avg:.4f} '
                f'Time={elapsed:.1f}s'
            )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_meter.avg,
        'cl_loss': cl_meter.avg,
    }


@torch.no_grad()
def validate(model, loader, device, threshold=0.5, debug=False):
    """验证/测试"""
    model.eval()
    metrics = ChangeDetectionMetrics(threshold=threshold)

    for step, batch in enumerate(loader):
        if debug and step >= 5:
            break

        img_t = batch['img_t'].to(device, non_blocking=True)
        img_t1 = batch['img_t1'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        pred = model(img_t, img_t1)  # [B, 1, H, W]
        metrics.update(pred, label)

    return metrics.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机种子
    seed = args.seed or cfg['train'].get('seed', 42)
    set_random_seed(seed)

    # 输出目录
    save_dir = cfg['output']['save_dir']
    log_dir = cfg['output']['log_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Logger
    logger = setup_logger(log_dir)
    logger.info(f'Config: {args.config}')
    logger.info(f'Device: {device}')
    logger.info(f'Seed: {seed}')

    # ---- 数据加载 ----
    ds_cfg = cfg['dataset']
    train_loader = build_dataloader(
        dataset_name=ds_cfg['name'],
        root=ds_cfg['root'],
        split='train',
        img_size=ds_cfg['img_size'],
        batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'],
    )
    val_loader = build_dataloader(
        dataset_name=ds_cfg['name'],
        root=ds_cfg['root'],
        split='val',
        img_size=ds_cfg['img_size'],
        batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'],
    )

    # ---- 模型 ----
    model_cfg = cfg['model']
    model = build_model(model_cfg).to(device)

    # 多 GPU
    if torch.cuda.device_count() > 1:
        logger.info(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    param_info = count_parameters(model)
    logger.info(f'Parameters: {param_info["total_M"]}M total, '
                f'{param_info["trainable_M"]}M trainable')

    # ---- 优化器 & 调度器 ----
    train_cfg = cfg['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'],
        eta_min=1e-6
    )
    scaler = GradScaler(enabled=train_cfg.get('amp', True))

    # ---- Checkpoint ----
    ckpt_manager = CheckpointManager(
        save_dir=save_dir,
        max_keep=cfg['output'].get('max_checkpoints', 5)
    )
    early_stopper = EarlyStopping(patience=30)

    start_epoch = 0
    best_f1 = 0.0

    # 恢复训练
    if args.resume:
        logger.info(f'Resuming from: {args.resume}')
        info = ckpt_manager.load(model, args.resume, optimizer, scheduler, str(device))
        start_epoch = info['epoch'] + 1
        best_f1 = info['metrics'].get('F1', 0.0)
        logger.info(f'Resumed from epoch {start_epoch}, best F1: {best_f1:.2f}%')

    # ---- 训练循环 ----
    logger.info('='*60)
    logger.info('Start Training')
    logger.info('='*60)

    for epoch in range(start_epoch, train_cfg['epochs']):
        logger.info(f'\n--- Epoch {epoch+1}/{train_cfg["epochs"]} ---')
        logger.info(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # 训练
        train_info = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch+1, logger,
            use_amp=train_cfg.get('amp', True),
            debug=args.debug
        )

        # 调度器更新
        scheduler.step()

        # 验证
        eval_freq = cfg['eval'].get('eval_freq', 1)
        if (epoch + 1) % eval_freq == 0:
            val_metrics = validate(
                model, val_loader, device,
                threshold=cfg['eval']['threshold'],
                debug=args.debug
            )
            logger.info(
                f'[Val] F1={val_metrics["F1"]:.2f}% | '
                f'IoU={val_metrics["IoU"]:.2f}% | '
                f'OA={val_metrics["OA"]:.2f}%'
            )

            # 判断是否为最佳
            is_best = val_metrics['F1'] > best_f1
            if is_best:
                best_f1 = val_metrics['F1']
                logger.info(f'✓ New best F1: {best_f1:.2f}%')

            # 保存 checkpoint
            ckpt_path = ckpt_manager.save(
                model if not isinstance(model, nn.DataParallel) else model.module,
                optimizer, scheduler,
                epoch=epoch+1,
                metrics=val_metrics,
                is_best=is_best
            )

            # 早停
            if early_stopper(val_metrics['F1']):
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    logger.info('='*60)
    logger.info(f'Training Complete. Best F1: {best_f1:.2f}%')
    logger.info(f'Best model saved to: {save_dir}/best_model.pth')
    logger.info('='*60)


if __name__ == '__main__':
    main()
