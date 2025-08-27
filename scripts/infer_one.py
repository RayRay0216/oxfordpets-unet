import argparse, torch
from PIL import Image
from torchvision import transforms as T
from models.unet import UNet
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    model = UNet().to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])(img)

    with torch.no_grad():
        logit = model(x.unsqueeze(0).to(args.device))
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(args.out)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
