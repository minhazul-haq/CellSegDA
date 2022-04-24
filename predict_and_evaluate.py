# author: Mohammad Minhazul Haq
# prediction and evaluation script

import numpy as np
import torch
from torch.utils import data
import os
from PIL import Image
import pickle
import argparse

from dataset.target_dataset_without_label import TargetUnlabelledDataSet
from model.unet_model import UNet
from utils import compute_iou, compute_dice_score

parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, default='tnbc', help='test dataset name: kirc or tnbc')
parser.add_argument('--model_path', type=str, default='saved_models/cellseg_uda_kirc_tnbc/best_model_5000.pth',
                    help='path to best model')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()

TEST_DATASET = args.test_dataset
RESTORE_FROM = args.model_path
GPU_DEVICE = args.gpu
EXPERIMENT_NAME = RESTORE_FROM.split('/')[1]

TEST_DIR = os.path.join('data', 'target', TEST_DATASET)
MEAN_STD_FILE = os.path.join('data', 'target', TEST_DATASET, TEST_DATASET + '_mean_std.txt')
PRED_DIR = os.path.join('predictions', EXPERIMENT_NAME)
GT_DIR = os.path.join('data', 'target', TEST_DATASET, 'test_masks')


def predict():
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)

    model = UNet(n_channels=3, n_classes=1)

    if TEST_DATASET == 'kirc':
        image_extension = '.tiff'
    elif TEST_DATASET == 'tnbc':
        image_extension = '.png'

    saved_state_dict = torch.load(RESTORE_FROM)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.to(GPU_DEVICE)

    with open(MEAN_STD_FILE, 'rb') as handle:
        data_mean_std = pickle.loads(handle.read())

    mean_val = data_mean_std['mean_val_images']
    std_val = data_mean_std['std_val_images']

    test_loader = data.DataLoader(TargetUnlabelledDataSet(root_dir=TEST_DIR,
                                                          image_folder='test_images',
                                                          mean=mean_val,
                                                          std=std_val),
                                  batch_size=1,
                                  shuffle=False)

    for iter, batch in enumerate(test_loader):
        image, name = batch
        image = image.to(GPU_DEVICE)

        pred = model(image)
        pred = pred.data.cpu().numpy()
        pred = pred.squeeze()

        pred_binarized = np.zeros_like(pred)
        threshold = 0.5
        pred_binarized[pred > threshold] = 255

        output_image = Image.fromarray(pred_binarized.astype(np.uint8))
        output_image.save('%s/%s' % (PRED_DIR, name[0].replace('_image' + image_extension, '_mask.png')))

        print('%d processed' % (iter + 1))


def evaluate():
    iou_scores = []
    dice_scores = []

    gt_filenames = sorted(os.listdir(GT_DIR))

    for gt_filename in gt_filenames:
        gt_image = Image.open(os.path.join(GT_DIR, gt_filename)).convert('1')
        gt = np.array(gt_image).astype('uint8')

        pred_filename = gt_filename
        pred = np.array(Image.open(os.path.join(PRED_DIR, pred_filename)).convert('1')).astype('uint8')

        iou_score = compute_iou(gt, pred)
        iou_scores.append(iou_score)

        dice_score = compute_dice_score(gt, pred)
        dice_scores.append(dice_score)

        print(gt_filename + ': iou ' + str(iou_score) + ', dice score ' + str(dice_score))

    print('iou: ' + str(np.mean(iou_scores)))
    print('dice score: ' + str(np.mean(dice_scores)))


if __name__ == '__main__':
    predict()
    evaluate()
