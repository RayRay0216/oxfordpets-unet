import argparse, os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.unet import UNet
from data import make_loaders
from utils import bce_dice_loss, dice_metric, iou_metric, set_seed


def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        opt.zero_grad(set_to_none=True)
        if scaler is None:
            logits = model(imgs)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = bce_dice_loss(logits, masks)
        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_metric(logits, masks) * imgs.size(0)
        total_iou += iou_metric(logits, masks) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/.torch/datasets")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--early-patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        root=os.path.expanduser(args.data_root),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=2,
    )

    model = UNet(in_ch=3, n_classes=1, base_ch=32).to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and "cuda" in args.device) else None

    best_dice, patience = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, args.device, scaler)
        va_loss, va_dice, va_iou = evaluate(model, val_loader, args.device)
        sched.step()

        print(
            f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} "
            f"| dice {va_dice:.4f} | IoU {va_iou:.4f}"
        )

        # early stopping + checkpoint
        if va_dice > best_dice:
            best_dice = va_dice
            patience = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                os.path.join(args.outdir, "best.pt"),
            )
        else:
            patience += 1
            if patience >= args.early_patience:
                print("Early stopping triggered.")
                break

    # final test
    ckpt = torch.load(os.path.join(args.outdir, "best.pt"), map_location=args.device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_dice, te_iou = evaluate(model, test_loader, args.device)
    print(f"Test | loss {te_loss:.4f} | dice {te_dice:.4f} | IoU {te_iou:.4f}")


if __name__ == "__main__":
    main()
