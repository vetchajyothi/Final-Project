import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Brain CT scan images and their exactly matching
    segmentation masks. Used for training the U-Net CNN for clot/lesion detection.
    """
    def __init__(self, images_dir: str, masks_dir: str, image_size=(256, 256)):
        """
        Args:
            images_dir (str): Path to folder containing original CT scan images.
            masks_dir (str): Path to folder containing binary mask images (white for clot, black for bg).
            image_size (tuple): Output size of tensors for U-Net. Default is 256x256.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # Ensure the directories actually exist
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Cannot find directories:\nImages: {images_dir}\nMasks: {masks_dir}")

        # List all image files. We assume masks have the exact same filename.
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
        # Transforms for the IMAGE
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(), # converts 0-255 to 0.0-1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transforms for the MASK
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        img_path = os.path.join(self.images_dir, img_name)
        # Assuming the mask has the exact same name as the original image
        mask_path = os.path.join(self.masks_dir, img_name) 

        # Load image and convert to RGB (U-Net expects n_channels=3)
        image = Image.open(img_path).convert("RGB")
        
        # Load mask and convert to grayscale (1 channel)
        mask = Image.open(mask_path).convert("L")
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        
        # Binarize mask exactly to 0.0 and 1.0 (in case of aliasing during resize)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
