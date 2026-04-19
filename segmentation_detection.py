import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Standard U-Net CNN Architecture for Semantic Image Segmentation.
    Outputs a pixel-wise mask identifying exactly where the clot/lesion is.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Upsampling path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        
        # Use Sigmoid to output probabilities between 0 and 1
        return torch.sigmoid(logits)

def extract_clots_from_mask(mask_prob: np.ndarray, threshold: float = 0.5):
    """
    Takes the CNN output probability mask and finds separate clots.
    Returns the number of clots, total damaged area in pixels, and the bounding boxes/contours.
    """
    # Threshold the mask
    binary_mask = (mask_prob > threshold).astype(np.uint8) * 255
    
    # Prevent border artifacts hooking the entire image frame
    binary_mask[0, :] = 0
    binary_mask[-1, :] = 0
    binary_mask[:, 0] = 0
    binary_mask[:, -1] = 0
    
    # Find contours using OpenCV
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    num_clots = 0
    clot_areas = []
    valid_contours = []
    
    total_pixels = mask_prob.size
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignore extremely tiny artifacts (noise) AND ignore massive background detections
        if area > 10.0 and area < (total_pixels * 0.3):  
            num_clots += 1
            clot_areas.append(int(area))
            valid_contours.append(cnt)
            
    return num_clots, clot_areas, valid_contours, binary_mask
