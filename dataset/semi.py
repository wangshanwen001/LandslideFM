from dataset.transform import *

from copy import deepcopy
import csv
import math
import numpy as np
import os
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ENV_FIELDS = ["landslide_exists", "has_building", "has_road",
              "has_water", "has_vegetation"]
NUM_ENV_FIELDS = len(ENV_FIELDS)


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None,
                 env_csv_path=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            # ↓ 修复：val 模式下，若传入了 id_path 则优先使用（用于分区评估）
            if id_path is not None:
                with open(id_path, 'r') as f:
                    self.ids = f.read().splitlines()
            else:
                with open('splits/%s/val.txt' % name, 'r') as f:
                    self.ids = f.read().splitlines()

        self.env_labels = None
        if env_csv_path is not None and mode == 'train_l':
            self.env_labels = self._load_env_csv(env_csv_path)

    def _load_env_csv(self, csv_path):
        """Load env labels from CSV into a dict: {image_name: tensor}."""
        import torch as _torch
        labels = {}
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [h.strip().lstrip("\ufeff") for h in reader.fieldnames]
            for r in reader:
                name = (r.get("image_name") or r.get("image") or "").strip()
                if not name or name.startswith("#"):
                    continue
                vec = []
                for field in ENV_FIELDS:
                    v = (r.get(field) or "").strip()
                    vec.append(1.0 if v.lower() in ("1", "true", "yes", "y", "t") else 0.0)
                labels[name] = _torch.tensor(vec, dtype=_torch.float32)
        return labels

    def _get_env_label(self, id_str):
        """Look up env label for an image. Returns dummy -1 tensor if not found."""
        parts = id_str.split()
        img_name = os.path.basename(parts[0])
        if self.env_labels is not None and img_name in self.env_labels:
            return self.env_labels[img_name]
        return torch.full((NUM_ENV_FIELDS,), -1.0, dtype=torch.float32)

    def __getitem__(self, item):
        id = self.ids[item]
        parts = id.split()

        img = Image.open(os.path.join(self.root, parts[0])).convert('RGB')

        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))

        elif len(parts) == 1:
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))

        else:
            mask = Image.fromarray(
                np.array(Image.open(os.path.join(self.root, parts[1])))
            )
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            env_label = self._get_env_label(id)
            return img, mask, env_label
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
