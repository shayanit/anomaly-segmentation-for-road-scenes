import torch
import torch.nn as nn
import torchvision.models as models

def resnet50SegmentationModel(model_state_dict = [], num_classes = 20):
    model = models.resnet50(pretrained=False)
    if (model_state_dict):
        model.load_state_dict(model_state_dict, strict=False)
    # for param in model.parameters():
    #     param.requires_grad = False #FREEZE
    model = nn.Sequential(*list(model.children())[:-2])
    segmentation_head = nn.Conv2d(2048, num_classes, kernel_size=1)
    upsample = nn.Upsample(size=(224,224), mode='bilinear', align_corners=False)
    final_model = nn.Sequential(model, segmentation_head, upsample)
    return final_model