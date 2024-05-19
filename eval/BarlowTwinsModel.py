import torch
import torch.nn as nn
import torchvision.models as models

barlowtwins_resnet = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
# for param in barlowtwins_resnet.parameters():
#     param.requires_grad = False
barlowtwins_resnet = nn.Sequential(*list(barlowtwins_resnet.children())[:-2])
num_classes = 20
segmentation_head = nn.Conv2d(2048, num_classes, kernel_size=1)
upsample = nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
BarlowTwinsModel = nn.Sequential(barlowtwins_resnet, segmentation_head, upsample)