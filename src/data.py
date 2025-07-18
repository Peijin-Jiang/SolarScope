import os
import torch
import random
import numpy as np
import pandas as pd

from typing import List
from PIL import Image
import torchvision.transforms.functional as F
from monai import transforms as monai_transforms
from torch.utils.data import Dataset, DataLoader
import time


def augment_image_and_mask(image, mask, resize_size=1024, max_translate=128):
    """
    Apply standard augmentations to an image and its corresponding mask.
    
    Args:
    - image (PIL.Image): The input image.
    - mask (PIL.Image): The input mask.
    - resize_size (int): The target size for resizing (height, width).
    
    Returns:
    - image (Tensor): The augmented image as a tensor.
    - mask (Tensor): The augmented mask as a tensor.
    """
    if random.random() > 0.2:
        image = F.hflip(image)
        mask = F.hflip(mask)
    if random.random() > 0.2:
        image = F.vflip(image)
        mask = F.vflip(mask)
    if random.random() > 0.2:
        angle = random.uniform(-30, 30)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)  # NEAREST interpolation for masks
    if random.random() > 0.2:
        translate_x = random.uniform(-max_translate, max_translate)
        translate_y = random.uniform(-max_translate, max_translate)
        image = F.affine(image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)
        mask = F.affine(mask, angle=0, translate=(translate_x, translate_y), scale=1, shear=0, interpolation=F.InterpolationMode.NEAREST)

    image = F.resize(image, (resize_size, resize_size))
    mask = F.resize(mask, (resize_size, resize_size), interpolation=F.InterpolationMode.NEAREST)
    mask = mask.long()  # Convert mask to long type for loss functions
    
    return image, mask


def get_bounding_box(image_size=1024, get_empty: bool = True) -> List[int]:
    W, H = image_size, image_size
    if get_empty:  # everything belongs to one class, or test
        return [0, 0, W, H]
    else:
      raise NotImplementedError


class SegmentationDataset(Dataset):
    def __init__(self, 
                data_csv_root: str,
                image_size: int=1024,
                mask_size: int=256,
                ignore_index: int=255,
                model_name="sam",
                normalize: bool=True,
                transform=None,
                augmentation=False,
                training_ratio=1.0, # [0.0, 1.0], 1.0 full set
                folder_names=None):
        """This class is used to load the segmentation dataset"""
        # load the dataset csv
        self.dataset_csv = pd.read_csv(data_csv_root)
        # subsample the training ratio
        if training_ratio < 1.0:
          self.dataset_csv = self.dataset_csv.sample(frac=training_ratio, random_state=42).reset_index(drop=True)
        self.img_paths, self.mask_paths = self.load_image_mask_paths()
        # set the ignore index
        self.ignore_index = ignore_index
        # set the normalization
        self.normalize = normalize
        # set the transform
        self.transform = transform
        # set the image and mask size
        self.image_size, self.mask_size = image_size, mask_size
        # set augmentation
        self.augmentation = augmentation
        # model name
        self.model_name = model_name

    def load_image_mask_paths(self):
      image_paths, masks = [], []
      for _, row in self.dataset_csv.iterrows():
        image_paths.append(row.image_path)
        masks.append(row.mask_path)
      return image_paths, masks

    def __len__(self):
        return len(self.img_paths)

    def __get_one_item__(self, idx):

        img_path, mask_path = self.img_paths[idx], self.mask_paths[idx]
        original_image = torch.tensor(np.array(Image.open(img_path).convert('RGB')))
        original_image = original_image.permute(2,0,1)
        original_mask = torch.tensor(np.array(Image.open(mask_path).convert('L'))).unsqueeze(0)

        if self.augmentation:
          original_image, original_mask = augment_image_and_mask(original_image, original_mask)
        assert len(original_mask.shape) == 3

        resize_func = monai_transforms.Compose([
            monai_transforms.Resized(keys=['image'], spatial_size=(self.image_size, self.image_size), mode=['bilinear']),
            monai_transforms.Resized(keys=['mask'], spatial_size=(self.mask_size, self.mask_size), mode=['nearest']),
        ])
        transformed = resize_func({"image": original_image, "mask": original_mask})
        image, target = transformed['image'].squeeze(0), transformed['mask'].long().squeeze(0)
        
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)
        target = (target > 120).long()

        # assert target.max() <= 1
        prompt = get_bounding_box(1024)
  
        if "sam" in self.model_name:
          input_dict = self.transform(image, input_boxes=[[prompt]], return_tensors="pt")
          input_dict = {k: v.squeeze(0) for k, v in input_dict.items()}
          input_dict["ground_truth_mask"] = target
        elif "deeplab" in self.model_name or "unet" in self.model_name:
          image = self.transform(image, return_tensors="pt")
          input_dict = {"pixel_values": torch.from_numpy(image).permute(2, 0, 1).float(), "ground_truth_mask": target}
        elif "segformer" in self.model_name:
          image = self.transform(image, return_tensors="pt").pixel_values.squeeze()
          input_dict = {"pixel_values": image, "ground_truth_mask": target}
        else:
          raise NotImplementedError
        return input_dict
    
    def __getitem__(self, idx):
        for _ in range(1000000):
          try:
            return self.__get_one_item__(idx)
          except Exception as e:
            print(e)
            time.sleep(1)
            idx = np.random.choice(len(self))