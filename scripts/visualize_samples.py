import os, argparse, torch, sys
import numpy as np
from PIL import Image

# 加入專案根目錄到 sys.path（避免從 scripts/ 執行時找不到 models/）
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torchvision import transforms as T
from models.unet import UNet
from data import BinaryOxfordPets

def overlay_mask(img_pil, mask_np, alpha=0.45):
    # img_pil: RGB, mask_np: {0,1} HxW
    img = np.array(img_pil).astype(np.float32)
    overlay = img.copy()
    color = np.array([255, 0, 0], dtype=np.float32)  # 紅色標示前景
    overlay[mask_np == 1] = color
    blended = (alpha * overlay + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(blended)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", default="viz")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--data-root", default="~/.torch/datasets")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds = BinaryOxfordPets(
        root=os.path.expanduser(args.data_root),
        split="test", img_size=args.img_size, aug=False, download=True
    )

    model = UNet().to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    saved = 0
    for i in range(len(ds)):
        if saved >= args.num: break
        img, gt = ds[i]  # img: [3,H,W], gt: [1,H,W] in {0,1}
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(args.device))
            prob = torch.sigmoid(logit)[0,0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)

        img_pil = T.ToPILImage()(img)
        mask_pil = Image.fromarray(pred * 255)
        over_pil = overlay_mask(img_pil, pred, alpha=0.45)

        img_pil.save(os.path.join(args.outdir, f"{i:04d}_img.jpg"))
        mask_pil.save(os.path.join(args.outdir, f"{i:04d}_pred.png"))
        over_pil.save(os.path.join(args.outdir, f"{i:04d}_overlay.jpg"))
        saved += 1

    print(f"Saved {saved} samples to {args.outdir}")

if __name__ == "__main__":
    main()
