import torch
import torch.nn as nn
import torch.nn.functional as F 

# Define Barlow Twins model
class BarlowTwinsModel(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer
        self.fc = nn.Conv2d(1000, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        # Reshape to add spatial dimensions
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add two singleton dimensions at the end
        # print(x.shape)
        x = self.fc(x)
        # Upsample the output tensor to match the original image size
        x = nn.functional.interpolate(x, size=(224, 448), mode='bilinear', align_corners=False)
        return x