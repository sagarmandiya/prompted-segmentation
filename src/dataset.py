import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PromptedSegmentationDataset(Dataset):
    """
    PyTorch Dataset for prompted segmentation with CLIPSeg.
    Loads images and binary masks with text prompts.
    """
    
    def __init__(self, split_json_path, prompt_text, transform=None, augment=False):
        """Initialize dataset with JSON split file and prompt."""
        with open(split_json_path, 'r') as f:
            self.data = json.load(f)
        
        self.prompt_text = prompt_text
        self.transform = transform
        self.augment = augment
        
        print(f"Loaded {len(self.data)} samples from {split_json_path} with prompt '{prompt_text}'")
        
        if self.augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Resize(640, 640, interpolation=cv2.INTER_LINEAR),
            ])
        else:
            self.aug_transform = A.Compose([
                A.Resize(640, 640, interpolation=cv2.INTER_LINEAR),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        if not os.path.exists(entry['image_path']):
            raise FileNotFoundError(f"Image not found: {entry['image_path']}")
        if not os.path.exists(entry['mask_path']):
            raise FileNotFoundError(f"Mask not found: {entry['mask_path']}")
        
        image = cv2.imread(entry['image_path'])
        if image is None:
            raise ValueError(f"Failed to load image: {entry['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {entry['mask_path']}")
        mask = (mask > 127).astype(np.uint8)
        
        augmented = self.aug_transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).float()
        
        return {
            'image': image,
            'mask': mask,
            'prompt': self.prompt_text,
            'image_path': entry['image_path'],
            'mask_path': entry['mask_path']
        }
    
    

def get_clipseg_transforms(processor):
    """Get transforms compatible with CLIPSeg processor."""
    def transform(image):
        return processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    return transform


