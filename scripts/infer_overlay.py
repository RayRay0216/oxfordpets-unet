import os, sys, argparse, torch, numpy as np
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torchvision import transforms as T
from models.unet import UNet

def overlay_mask(img_pil, mask_np, alpha=0.45):
    img = np.array(img_pil).astype(np.float32)
    overlay = img.copy()
    color = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask_np == 1] = color
    return Image.fromarray((alpha*overlay + (1-alpha)*img).astype(np.uint8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = UNet().to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"]); model.eval()

    img = Image.open(args.image).convert("RGB")
    x = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])(img)

    with torch.no_grad():
        logit = model(x.unsqueeze(0).to(args.device))
        prob = torch.sigmoid(logit)[0,0].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)

    overlay = overlay_mask(img.resize((args.img_size,args.img_size)), pred, alpha=0.45)
    overlay.save(args.out)
    print("Saved overlay to", args.out)

if __name__ == "__main__":
    main()
