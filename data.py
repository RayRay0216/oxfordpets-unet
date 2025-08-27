import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image


class BinaryOxfordPets(Dataset):
    """
    將 Oxford-IIIT Pet 的 segmentation 轉成二分類遮罩：pet(1) vs background(0)。
    官方標註值：1=pet, 2=border, 3=background。這裡把 border 視為背景。
    """
    def __init__(self, root, split="trainval", img_size=256, aug=True, download=True):
        self.ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.img_size = img_size
        self.aug = aug and (split != "test")

        self.resize_img = T.Resize((img_size, img_size), antialias=True)
        self.to_tensor = T.ToTensor()
        self.color = T.ColorJitter(0.2, 0.2, 0.2, 0.1)

    def __len__(self):
        return len(self.ds)

    def _rand_resized_crop(self, img, mask_pil):
        # img: PIL RGB, mask_pil: PIL L (values {0,1})
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.7, 1.0), ratio=(0.9, 1.1)
        )
        img = TF.resized_crop(
            img, i, j, h, w, size=(self.img_size, self.img_size), antialias=True
        )
        mask_pil = TF.resized_crop(
            mask_pil, i, j, h, w, size=(self.img_size, self.img_size),
            interpolation=TF.InterpolationMode.NEAREST
        )
        return img, mask_pil

    def __getitem__(self, idx):
        img, mask = self.ds[idx]  # img: PIL RGB, mask: PIL (Oxford mask, labels {1,2,3})

        # --- 將 Oxford mask 二值化為 {0,1} --- #
        # 官方: 1=pet, 2=border, 3=background；我們把 border 視為背景
        mask_np = (np.array(mask, dtype=np.uint8) == 1).astype(np.uint8)  # 0/1
        # 用 PIL 'L' 單通道承載 0/1（之後幾何增強用最近鄰避免插值污染）
        mask_pil = Image.fromarray(mask_np, mode="L")

        if self.aug:
            # 水平翻轉
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_pil = TF.hflip(mask_pil)
            # 顏色抖動（只對影像）
            if random.random() < 0.5:
                img = self.color(img)
            # 隨機 resize-crop
            if random.random() < 0.5:
                img, mask_pil = self._rand_resized_crop(img, mask_pil)
            else:
                img = self.resize_img(img)
                mask_pil = TF.resize(
                    mask_pil, (self.img_size, self.img_size),
                    interpolation=TF.InterpolationMode.NEAREST
                )
        else:
            img = self.resize_img(img)
            mask_pil = TF.resize(
                mask_pil, (self.img_size, self.img_size),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # 轉 tensor：影像是 [0,1] 浮點；遮罩是 {0,1} 浮點、帶 channel 維度
        img = self.to_tensor(img)  # float32 in [0,1]
        mask_t = torch.from_numpy(np.array(mask_pil, dtype=np.uint8)).unsqueeze(0).float()  # (1,H,W) in {0,1}

        return img, mask_t


def make_loaders(
    root, img_size=256, batch_size=8, num_workers=2, val_ratio=0.1, download=True
):
    full = BinaryOxfordPets(root, split="trainval", img_size=img_size, aug=True, download=download)
    n_val = max(1, int(len(full) * val_ratio))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    test_set = BinaryOxfordPets(root, split="test", img_size=img_size, aug=False, download=download)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader