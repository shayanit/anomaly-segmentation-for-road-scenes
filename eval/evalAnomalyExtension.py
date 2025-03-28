# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from enet import ENet
from bisenetv1 import BiSeNetV1
import sys
from torchvision.transforms import Resize
import torch.nn.functional as F
from enet_utils import load_checkpoint
import torch.optim as optim
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from Resnet50SegmentationModel import resnet50SegmentationModel


input_transform = Compose(
    [
        Resize((224, 224), Image.BILINEAR),
        ToTensor(),
    ]
)

label_transform = Compose(
    [
        Resize((224, 224), Image.BILINEAR),
    ]
)

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def min_max_scale(logits, min_scale=-1, max_scale=1):
    min_logit = torch.min(logits)
    max_logit = torch.max(logits)
    scaled_logits = (logits - min_logit) / (max_logit - min_logit) * (max_scale - min_scale) + min_scale
    return scaled_logits

def max_logit_normalized(logits):
    max_logits, _ = torch.max(logits, dim=0)
    return min_max_scale(max_logits)

def intersection_over_union(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Maximum Softmax Probability
def max_softmax(logits, temperature):
    softmax_probs = F.softmax(logits / temperature, dim=0)
    max_prob, _ = torch.max(softmax_probs, dim=0) # choose the highest maximum probability among all the class probabilities
    return max_prob 

# Entropy formula: -sum(p_i * log(p_i)) / log(num_classes)
# Entropy measures the uncertainty or disorder in a probability distribution.
def max_entropy_normalized(logits, epsilon=1e-10):
    assert len(logits.shape) == 3
    probs = F.softmax(logits, dim=0)
    probs = probs + epsilon
    entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=0), torch.log(torch.tensor(probs.shape[0], dtype=torch.float32)))
    return min_max_scale(entropy, min_scale=0, max_scale=1)
    # return entropy

def compute_pathGT(path):
    pathGT = path.replace('images', 'labels_masks')
    if 'RoadObsticle21' in pathGT:
        pathGT = pathGT.replace('webp', 'png')
    if 'fs_static' in pathGT:
        pathGT = pathGT.replace('jpg', 'png')
    if 'RoadAnomaly' in pathGT:
        pathGT = pathGT.replace('jpg', 'png')
    return pathGT
    
def compute_ood_gts(ood_gts, pathGT):
        if 'RoadAnomaly' in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if 'LostAndFound' in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if 'Streethazard' in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)
        return ood_gts

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith('module.'):
                own_state[name.split('module.')[-1]].copy_(param)
            else:
                print(name, ' not loaded')
                continue
        else:
            own_state[name].copy_(param)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="/content/datasets/Validation_Dataset/RoadAnomaly21/images/*.png")  
    parser.add_argument('--loadDir',default="/content/drive/MyDrive/datasets/")
    parser.add_argument('--loadWeights', default="barlow-twins.pth")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--method', default="msp")
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--q', action='store_true')
    parser.add_argument('--showimages', action='store_true')
    # parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    dataset = args.input
    dataset = dataset.split("/")[-3]
    evaluation_props = f"{args.loadWeights}\t{args.method}\t{dataset[0:15]}\tt={args.temperature}"
    if not args.q:
        print(f"Evaluating - {evaluation_props}")
    
    anomaly_score_list = np.array([])
    ood_gts_list = np.array([])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if (args.loadWeights==""):
        model_state_dict=[]
        print("Loading resnet50 model without weights!") if not args.q else ''
    else:
        weightspath = args.loadDir + args.loadWeights
        checkpoint = torch.load(weightspath)
        model_state_dict = checkpoint['model_state_dict']
        print(f"Loading resnet50 model with saved weights: {weightspath}") if not args.q else ''

    # Create model and load state dict
    model = resnet50SegmentationModel(model_state_dict)
    model.to(device)
    model.eval()
    
    validation_images = glob.glob(os.path.expanduser(str(args.input)))

    for path in validation_images:
        print(path) if not args.q else ''
        images = (Image.open(path).convert('RGB'))
        
        if args.showimages:
            plt.imshow(images)
            plt.axis('off')
            plt.show() 
        
        images = input_transform(images).unsqueeze(0).float().to(device)
        with torch.no_grad():
            model_output = model(images)
        
        # print(model_output)
        if (args.showimages):
            tensor = model_output

            # Remove the singleton dimension at the start
            tensor = tensor.squeeze(0)  # Now shape is [20, height, width]

            # Get the dimensions from the tensor
            num_classes, height, width = tensor.shape

            # Get the predicted class for each pixel
            predicted_classes = tensor.argmax(dim=0).cpu().numpy()  # Shape [height, width]

            # Define a color map for the classes
            colors = plt.cm.get_cmap('tab20', num_classes)

            # Create an RGB image to visualize the segmentation
            segmentation_image = np.zeros((height, width, 3), dtype=np.float32)

            # Map each class to its corresponding color
            for class_index in range(num_classes):
                mask = predicted_classes == class_index
                color = colors(class_index)[:3]  # Get RGB color

                # Assign color to pixels belonging to the current class
                segmentation_image[mask] = color

            # Display the segmentation image
            # plt.figure(figsize=(10, 10))
            plt.imshow(segmentation_image)
            plt.axis('off')
            # plt.title('Segmentation Map')
            plt.show()

        # Compute anomaly_result based on the method
        print(f"model_output result shape is: {model_output.shape}") if not args.q else ''
        result = model_output.squeeze(0)
        print(f"after squeeze(0): {result.shape}") if not args.q else ''
        # print(f"unique values in model output: {torch.unique(result.flatten())}")
        trimmed_result = result[:-1]
        print(f"after trim: {trimmed_result.shape}") if not args.q else ''
        if args.method == 'maxlogit':
            # Minimum logit is most anomalous as it's least confident
            anomaly_result = -1 * max_logit_normalized(trimmed_result)
        elif args.method == 'maxentropy':
            anomaly_result = max_entropy_normalized(trimmed_result)
        elif args.method == 'msp':
            anomaly_result = 1 - max_softmax(trimmed_result, args.temperature) # so that higher anomaly scores correspond to higher likelihoods of being anomalous
        elif args.method == 'void':
            softmax_probs = F.softmax(result, dim=0)
            anomaly_result = softmax_probs[-1]
        else:
            sys.exit("No method argument is defined.")
        # print(f"anomaly result shape: {anomaly_result.shape}")

        if args.showimages:
            plt.imshow(min_max_scale(anomaly_result.cpu()), cmap="coolwarm", interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.show() 
        
        pathGT = compute_pathGT(path)

        mask = Image.open(pathGT)
        mask = label_transform(mask)
        np_mask = np.array(mask)

        # out-of-distribution ground truths
        ood_gts = compute_ood_gts(np_mask, pathGT)
        if args.showimages:
          # Create an RGB image where all values are initially set to blue
          # Shape will be (w, h, 3) for RGB
          height, width = ood_gts.shape
          l_image = np.zeros((height, width, 3))

          # Set red color where ood_gts is 1
          l_image[ood_gts == 1] = [1, 0, 0]  # Red
          # Set blue color where ood_gts is 0
          l_image[ood_gts != 1] = [0, 0, 1]  # Blue

          # Display the image
          plt.imshow(l_image)
          plt.axis('off')  # Hide the axis
          plt.show()
        print(f"ood_gts shape: {ood_gts.shape}") if not args.q else ''

        if 1 not in np.unique(ood_gts):
            # no pixels labeled as anomaly
            # continue to the next iteration
            continue              
        else:
             anomaly_result = anomaly_result.data.cpu().numpy()
             print(f"anomaly_result shape: {anomaly_result.shape}") if not args.q else ''
            #  ood_gts_list.append(ood_gts)
            #  anomaly_score_list.append(anomaly_result)
             ood_gts_list = np.append(ood_gts_list, ood_gts)
             anomaly_score_list = np.append(anomaly_score_list, anomaly_result)
        del result, anomaly_result, ood_gts, mask, np_mask
        torch.cuda.empty_cache()

    # ood_gts = np.array(ood_gts_list)
    # anomaly_scores = np.array(anomaly_score_list)
    
    ood_gts = ood_gts_list
    anomaly_scores = anomaly_score_list

    # out-of-distribution regions
    ood_mask = (ood_gts == 1)
    #  in-distribution regions
    ind_mask = (ood_gts == 0)

    # print(f"anomaly_scores size: {anomaly_scores.shape}")
    # print(f"ood_mask size: {ood_mask.shape}")

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    
    # a = np.max(val_out)
    # b = np.min(val_out)

    
    prc_auc = average_precision_score(val_label, val_out) # precision=true_positives/total_positives, higher is better
    fpr95 = fpr_at_95_tpr(val_out, val_label) #false positive rate at 95% true positive, lower is better
    
    # from sklearn.metrics import roc_curve, roc_auc_score

    # Compute ROC curve and AUC
    # fpr, tpr, thresholds = roc_curve(val_label, val_out)

    # Find optimal threshold based on the ROC curve
    # optimal_threshold_index = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_threshold_index]
    
    # threshold = 0.5 
    # threshold = optimal_threshold
    
    if not args.q:
        if args.method == 'maxlogit':
            print("maxlogit method score values are normalized in the range [-1, 1]")
        elif args.method == 'maxentropy':
            print("maxentropy method score values are normalized in the range [0, 1]")
        elif args.method == 'msp':
            print("msp method score values are softmax in the range [-1, 1]")
        elif args.method == 'void':
            print("void classifier method score values are softmax in the range [-1, 1]")

    # print(f"optimal threshold is: {threshold}")
    
    # if score is bigger than treshold, it's an anomaly
    # prediction = (val_out > threshold).astype(int)
    # ground_truth = val_label
    
    # IoU = intersection_over_union(ground_truth, prediction)
    
    result_content = f"{evaluation_props}\tAUPRC score: {round(prc_auc*100.0,3)}\tFPR@TPR95: {round(fpr95*100.0,3)}"
    # , optimal_threshold: {optimal_threshold}, IoU: {IoU}
    print(result_content)
    
    result_path = "results.txt"
    if not os.path.exists(result_path):
        open(result_path, 'w').close()
    with open(result_path, 'a') as file:
        file.write(result_content)

if __name__ == '__main__':
    main()