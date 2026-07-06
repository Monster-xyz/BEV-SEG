"""Training pipeline for BEV parking segmentation."""

import os
import sys
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from dataset import build_dataloaders
from model import build_model, count_params

# Dice Loss: region overlap measure, complements CE per-pixel accuracy
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# Build combined loss (CE + Dice) based on config string
def build_loss(cfg):
    losses = []
    weights = []
    if "ce" in cfg.LOSS:
        losses.append(nn.CrossEntropyLoss(ignore_index=255))  # ignore=255 skips unlabeled boundary pixels
        weights.append(1.0)
    if "dice" in cfg.LOSS:
        losses.append(DiceLoss())
        weights.append(1.0)
    if not losses:
        losses.append(nn.CrossEntropyLoss(ignore_index=255))  # ignore=255 skips unlabeled boundary pixels
        weights.append(1.0)

    def combined_loss(logits, targets):
        total = 0.0
        for w, fn in zip(weights, losses):
            total += w * fn(logits, targets)
        return total
    return combined_loss

# @torch.no_grad(): prevents graph construction, essential for val/test
# Without it, GPU memory leaks silently.
@torch.no_grad()
def compute_iou(pred_logits, mask, num_classes):
    pred = pred_logits.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (mask == c)).sum().float()
        union = ((pred == c) | (mask == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
        else:
            ious.append(1.0)
    return ious

# Save full state (model+optimizer+scheduler+epoch) for exact resume
def save_checkpoint(epoch, model, optimizer, scheduler, best_iou, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_iou": best_iou,
    }, path)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    # Load to CPU first to avoid cross-device errors, then .to(device)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  [Checkpoint] Loaded from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("best_iou", 0.0)

# One epoch: forward -> backward -> (accumulate) -> param update
def train_epoch(model, loader, criterion, optimizer, scaler, epoch, writer, cfg):
    model.train()  # Enable BN/Dropout training behavior
    total_loss = 0.0
    num_batches = len(loader)
    start_time = time.time()
    device = next(model.parameters()).device

    for step, (images, masks) in enumerate(loader):
        # Async GPU transfer, CPU continues without waiting
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if scaler:
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
        else:
            logits = model(images)
            loss = criterion(logits, masks)

        # Divide for gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS
        loss = loss / cfg.ACCUM_STEPS

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.ACCUM_STEPS == 0:
            if cfg.GRAD_CLIP > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                # Clip gradients to prevent explosion from bad batches
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # set_to_none=True: faster than zero_grad(), PyTorch-recommended
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.ACCUM_STEPS

        if step % cfg.LOG_INTERVAL == 0:
            global_step = epoch * num_batches + step
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss", loss.item() * cfg.ACCUM_STEPS, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            elapsed = time.time() - start_time
            print(f"  E{epoch:02d} S{step:04d}/{num_batches}  "
                  f"loss={loss.item() * cfg.ACCUM_STEPS:.4f}  "
                  f"lr={lr:.2e}  time={elapsed:.0f}s")

    avg_loss = total_loss / num_batches
    return avg_loss

# model.eval() changes BN/Dropout behavior
# @torch.no_grad() disables autograd graph construction
# Both needed for correct validation.
@torch.no_grad()
def validate(model, loader, criterion, cfg, epoch, writer):
    model.eval()  # Disable Dropout, BN uses running stats
    total_loss = 0.0
    total_ious = [0.0] * cfg.NUM_CLASSES
    n_batches = len(loader)
    device = next(model.parameters()).device
    saved_viz = False

    for step, (images, masks) in enumerate(loader):
        # Async GPU transfer, CPU continues without waiting
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        ious = compute_iou(logits, masks, cfg.NUM_CLASSES)
        for c in range(cfg.NUM_CLASSES):
            total_ious[c] += ious[c]

        if not saved_viz and writer is not None:
            pred = logits.argmax(dim=1).cpu().numpy()
            img_grid = images[:4].cpu()
            mask_grid = masks[:4].cpu()
            pred_grid = torch.from_numpy(pred[:4])
            # Approximate de-normalization (reverse ImageNet std/mean) for display
            writer.add_images("val/images", (img_grid * 0.25 + 0.45).clamp(0, 1), epoch)
            writer.add_images("val/gt_masks",
                (mask_grid.unsqueeze(1).float() / (cfg.NUM_CLASSES - 1)).clamp(0, 1), epoch)
            writer.add_images("val/pred_masks",
                (pred_grid.unsqueeze(1).float() / (cfg.NUM_CLASSES - 1)).clamp(0, 1), epoch)
            saved_viz = True

    avg_loss = total_loss / n_batches
    avg_ious = [v / n_batches for v in total_ious]
    miou = float(np.mean(avg_ious))

    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)
        for c in range(cfg.NUM_CLASSES):
            writer.add_scalar(f"val/IoU_class_{c}", avg_ious[c], epoch)

    return avg_loss, avg_ious, miou

# Main: assemble -> data -> model -> optimizer -> loop -> checkpoint
# This structure is universal across DL training codebases.
def main():
    # Auto-fallback to CPU if no GPU, prevents crash on different machines
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    print("Building model...")
    model = build_model(cfg)
    model.to(device)
    print(f"  Parameters: {count_params(model):,}")

    criterion = build_loss(cfg)

    # AdamW: decoupled weight decay, current industry standard
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # Two scheduler options:
    # - cosine: smooth decay, good default
    # - plateau: reduce on val_loss stagnation
    if cfg.LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN,
        )
    elif cfg.LR_SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=cfg.LR_MIN,
        )
    else:
        scheduler = None

    # AMP mixed precision: FP16 forward/backward saves ~40% VRAM on GTX 1650
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("AMP enabled (mixed precision)")

    run_name = f"run_{time.strftime('%m%d_%H%M')}"
    log_dir = os.path.join(cfg.LOG_DIR, run_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log: {log_dir}")

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        clean_cfg = {k: v for k, v in vars(cfg).items()
                     if not k.startswith("_") and k.isupper()}
        json.dump(clean_cfg, f, indent=2, default=str)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    start_epoch = 0
    best_miou = 0.0
    # Resume from previous run: restore model + optimizer + scheduler state
    if cfg.RESUME_FROM and os.path.isfile(cfg.RESUME_FROM):
        start_epoch, best_miou = load_checkpoint(
            cfg.RESUME_FROM, model, optimizer,
            scheduler if cfg.LR_SCHEDULER == "cosine" else None,
        )
        start_epoch += 1

    print(f"\n{'='*50}")
    print(f"Training started  |  epochs={cfg.EPOCHS}  batch={cfg.BATCH_SIZE}  loss={cfg.LOSS}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, writer, cfg,
        )

    # Two scheduler options:
    # - cosine: smooth decay, good default
    # - plateau: reduce on val_loss stagnation
        if cfg.LR_SCHEDULER == "cosine":
            scheduler.step()

        val_loss, class_ious, miou = validate(
            model, val_loader, criterion, cfg, epoch, writer,
        )

        if cfg.LR_SCHEDULER == "plateau":
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        iou_str = "  ".join([f"Cls{c}={v:.3f}" for c, v in enumerate(class_ious)])
        print(f"\n--- Epoch {epoch:02d}/{cfg.EPOCHS-1}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"mIoU={miou:.4f}  "
              f"time={epoch_time:.0f}s")
        print(f"    Per-class IoU: {iou_str}")
        print(f"    LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        # best_model.pt = best val mIoU (for deployment)
        # epoch_xx.pt = periodic snapshot (for rollback)
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            save_checkpoint(epoch, model, optimizer, scheduler, best_miou,
                            os.path.join(cfg.CKPT_DIR, "best_model.pt"))
            print("  *** New best mIoU!")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_miou,
                            os.path.join(cfg.CKPT_DIR, f"epoch_{epoch:02d}.pt"))

    save_checkpoint(cfg.EPOCHS - 1, model, optimizer, scheduler, best_miou,
                    os.path.join(cfg.CKPT_DIR, "final_model.pt"))

    writer.close()
    print(f"\n{'='*50}")
    print(f"Training complete.  Best val mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {cfg.CKPT_DIR}")
    print(f"TensorBoard logs:     {log_dir}")
    print(f"{'='*50}")


# Entry guard: only runs when script is executed directly, not imported
if __name__ == "__main__":
    main()
    xjt = 22222
    zy = "damn"