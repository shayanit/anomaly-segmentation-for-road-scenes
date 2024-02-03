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

softmax = torch.nn.functional.softmax
log_softmax = torch.nn.functional.log_softmax

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

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--weightsDir',default="../trained_models/")
    # parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--method', default="")
    parser.add_argument('--classifier', default="pre-trained erfnet")
    # parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    anomaly_score_list = []
    ood_gts_list = []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modelpath = args.loadDir + args.loadModel
    weightspath = args.weightsDir + args.model

    print ("Loading model: " + args.model)
    print ("Loading weights: " + weightspath)

    # Create model and load state dict
    if args.model == "erfnet":
        model = ERFNet(NUM_CLASSES)
        model = torch.nn.DataParallel(model).to(device)
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    elif args.model == "bisenetv1":
        model = BiSeNetV1(NUM_CLASSES)
        model.aux_mode = 'eval' # aux_mode can be train, eval, pred
        model = torch.nn.DataParallel(model).to(device)
        model = load_my_state_dict(model, torch.load(weightspath))
    elif args.model == "enet":
        model = ENet(NUM_CLASSES)
        model = torch.nn.DataParallel(model).to(device)
        model = ENet(NUM_CLASSES)
        model = load_my_state_dict(model.module, torch.load(weightspath)['state_dict'])
    
    print ("Model and weights loaded successfully!")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        images = images.permute(0,3,1,2)
        
        with torch.no_grad():
            model_output = model(images)
            
        # Compute model result to be used
        if args.model == "erfnet":
            result = model_output.squeeze(0)
        elif args.model == "bisenetv1":
            result = model_output[0].squeeze(0)
        elif args.model == "enet":
            result = torch.roll(model_output, -1, 0)
        
        # Compute anomaly_result based on the method
        trimmed_result = result[:-1]
        if args.method == 'maxlogit':
            # Minimum logit is most anomalous
            anomaly_result, _ = -torch.max(trimmed_result, dim=0)
        elif args.method == 'maxentropy':
            # H(p) = -sum_{i=1}^{n} p_i * log(p_i)
            softmax_probs = softmax(trimmed_result, dim=0)
            log_softmax_probs = log_softmax(trimmed_result, dim=0)
            elementwise_product = -softmax_probs * log_softmax_probs
            pixelwise_entropy = torch.sum(elementwise_product, dim=0)
            num_classes = torch.tensor(trimmed_result.size(0))
            normalized_entropy = torch.div(pixelwise_entropy, torch.log(num_classes))
            anomaly_result = normalized_entropy
        elif args.method == 'msp':
            # Maximum Softmax Probability
            softmax_probs = softmax(trimmed_result / args.temperature, dim=0)
            max_prob, _ = torch.max(softmax_probs, dim=0)
            anomaly_result = 1.0 - max_prob
        elif args.classifier == 'void':
            softmax_probs = softmax(result, dim=0)
            anomaly_result = softmax_probs[-1]
        else:
            sys.exit("No method argument is defined.")
        
        pathGT = compute_pathGT(path)

        mask = Image.open(pathGT)
        np_mask = np.array(mask)

        # out-of-distribution ground truths
        ood_gts = compute_ood_gts(np_mask, pathGT)

        if 1 not in np.unique(ood_gts):
            # no pixels labeled as anomaly
            # continue to the next iteration
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_result = anomaly_result.data.cpu().numpy()
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask, np_mask
        torch.cuda.empty_cache()

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # out-of-distribution regions
    ood_mask = (ood_gts == 1)
    #  in-distribution regions
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()