# author: Mohammad Minhazul Haq
# training script for CellSegSSDA model

import torch
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import pickle
import argparse

from dataset.source_dataset import SourceDataSet
from dataset.target_dataset_with_partial_label import TargetPartialLabelledDataSet
from dataset.target_dataset_with_full_label import TargetLabelledDataSet
from model.unet_model import UNet
from model.unet_decoder_model import UNetDecoder
from model.discriminator_model import FCDiscriminator
from utils import compute_dice_score, compute_iou

parser = argparse.ArgumentParser()
parser.add_argument('--source_dataset', type=str, default='kirc', help='source dataset name: kirc or tnbc')
parser.add_argument('--target_dataset', type=str, default='tnbc', help='target dataset name: kirc or tnbc')
parser.add_argument('--target_label_percentage', type=int, default=25,
                    help='percentage of available target labels: 10, 25, 50, 75')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--steps', type=int, default=5000, help='number of steps')
parser.add_argument('--save_model_at', type=int, default=500, help='save model after steps')
args = parser.parse_args()

SOURCE_DATASET = args.source_dataset
TARGET_DATASET = args.target_dataset
TARGET_DATASET_DIR = TARGET_DATASET + '_ss_0.' + str(args.target_label_percentage)
EXPERIMENT_NAME = 'cellseg_ssda_' + SOURCE_DATASET + '_' + TARGET_DATASET
SAVED_MODEL_DIR = os.path.join('saved_models', EXPERIMENT_NAME)

SOURCE_DIR = os.path.join('data', 'source', SOURCE_DATASET)
SOURCE_MEAN_STD_FILE = os.path.join('data', 'source', SOURCE_DATASET, SOURCE_DATASET + '_mean_std.txt')
TARGET_DIR = os.path.join('data', 'target', TARGET_DATASET_DIR)
TARGET_MEAN_STD_FILE = os.path.join('data', 'target', TARGET_DATASET_DIR, TARGET_DATASET + '_mean_std.txt')

NUM_STEPS = args.steps
SAVE_MODEL_EVERY = args.save_model_at
BATCH_SIZE = args.batch_size
GPU = args.gpu

LEARNING_RATE_SEG = 0.0001
LEARNING_RATE_DIS = 0.001
LEARNING_RATE_DEC = 0.001
LAMBDA_ADV = 0.001
LAMBDA_RECONS = 0.01


def validate(model_seg, model_dis, model_dec, validation_loader, bce_logit_loss, reconstruction_loss):
    model_seg.eval()
    model_dis.eval()
    model_dec.eval()

    adv_target_losses = []
    rec_losses = []
    iou_scores = []
    dice_scores = []
    source_label = 0

    for index, batch in enumerate(validation_loader):
        image, label, name = batch
        image = Variable(image).cuda(GPU)

        pred = model_seg(image)
        output_dis = model_dis(pred)

        adv_target_loss = bce_logit_loss(output_dis,
                                         Variable(torch.FloatTensor(output_dis.data.size()).fill_(source_label)).cuda(
                                             GPU))
        adv_target_losses.append(adv_target_loss.data.cpu().numpy())

        reconstructed_image = model_dec(pred)
        recons_loss = reconstruction_loss(reconstructed_image, image)
        rec_losses.append(recons_loss.data.cpu().numpy())

        label = label.data.cpu().numpy()
        label = label.squeeze()
        label_binarized = np.zeros_like(label)
        label_binarized[label > 0] = 1

        pred = pred.data.cpu().numpy()
        pred = pred.squeeze()
        pred_binarized = np.zeros_like(pred)
        threshold = 0.5
        pred_binarized[pred > threshold] = 1

        iou_score = compute_iou(label, pred_binarized)
        iou_scores.append(iou_score)

        dice_score = compute_dice_score(label, pred_binarized)
        dice_scores.append(dice_score)

    return np.mean(adv_target_losses), np.mean(rec_losses), np.mean(iou_scores), np.mean(dice_scores)


def dice_coef_loss(y_pred, y_true):
    smooth = 1.

    iflat = y_pred.view(-1)
    tflat = y_true.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def train():
    cudnn.enabled = True

    # segmentation network
    model_seg = UNet(n_channels=3, n_classes=1)
    model_seg.train()
    model_seg.cuda(GPU)

    cudnn.benchmark = True

    # decoder network
    model_decoder = UNetDecoder(n_channels=1, n_classes=3)
    model_decoder.train()
    model_decoder.cuda(GPU)

    # discrimination network
    model_dis = FCDiscriminator(num_classes=1)
    model_dis.train()
    model_dis.cuda(GPU)

    if not os.path.exists(SAVED_MODEL_DIR):
        os.makedirs(SAVED_MODEL_DIR)

    max_iterations = NUM_STEPS * BATCH_SIZE

    # data normalization
    with open(SOURCE_MEAN_STD_FILE, 'rb') as handle:
        source_mean_std = pickle.loads(handle.read())

    mean_source_train = source_mean_std['mean_train_images']
    std_source_train = source_mean_std['std_train_images']

    source_loader = data.DataLoader(SourceDataSet(root_dir=SOURCE_DIR,
                                                  image_folder='train_images',
                                                  mask_folder='train_masks',
                                                  max_iters=max_iterations,
                                                  mean=mean_source_train,
                                                  std=std_source_train),
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)

    source_loader_iter = enumerate(source_loader)

    # data normalization
    with open(TARGET_MEAN_STD_FILE, 'rb') as handle:
        target_mean_std = pickle.loads(handle.read())

    mean_target_train = target_mean_std['mean_train_images']
    std_target_train = target_mean_std['std_train_images']

    mean_target_val = target_mean_std['mean_val_images']
    std_target_val = target_mean_std['std_val_images']

    target_train_loader = data.DataLoader(TargetPartialLabelledDataSet(root_dir=TARGET_DIR,
                                                                 image_folder='train_images',
                                                                 mask_folder='train_masks',
                                                                 max_iters=max_iterations,
                                                                 mean=mean_target_train,
                                                                 std=std_target_train),
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)

    target_train_loader_iter = enumerate(target_train_loader)

    target_val_loader = data.DataLoader(TargetLabelledDataSet(root_dir=TARGET_DIR,
                                                              image_folder='validation_images',
                                                              mask_folder='validation_masks',
                                                              mean=mean_target_val,
                                                              std=std_target_val),
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    # optimizers
    optimizer_seg = optim.Adam(model_seg.parameters(), lr=LEARNING_RATE_SEG)
    optimizer_seg.zero_grad()

    optimizer_dis = optim.Adam(model_dis.parameters(), lr=LEARNING_RATE_DIS)
    optimizer_dis.zero_grad()

    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=LEARNING_RATE_DEC)
    optimizer_decoder.zero_grad()

    bce_logit_loss = torch.nn.BCEWithLogitsLoss()
    reconstruction_loss = torch.nn.MSELoss()

    # labels for adversarial training
    source_label = 0
    target_label = 1

    best_val_iou_score = None

    for iter in range(1, NUM_STEPS + 1):
        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_recons_value = 0
        loss_dis_value = 0

        optimizer_seg.zero_grad()
        optimizer_dis.zero_grad()
        optimizer_decoder.zero_grad()

        _, batch_s = source_loader_iter.__next__()
        image_s, label_s, name_s = batch_s
        image_s = Variable(image_s).cuda(GPU)
        label_s = Variable(label_s).cuda(GPU)

        _, batch_t = target_train_loader_iter.__next__()
        image_t, label_exists_t, label_t, name_t = batch_t
        image_t = Variable(image_t).cuda(GPU)

        # train D

        for param in model_dis.parameters():
            param.requires_grad = True

        # train with source
        pred_s = model_seg(image_s)
        pred_s = pred_s.detach()
        dis_output_s = model_dis(pred_s)

        loss_dis_source = bce_logit_loss(dis_output_s,
                                         Variable(torch.FloatTensor(dis_output_s.data.size()).fill_(source_label)).cuda(GPU))
        loss_dis_source.backward()
        loss_dis_value += loss_dis_source.data.cpu().numpy()

        # train with target
        pred_t = model_seg(image_t)
        pred_t = pred_t.detach()
        dis_output_t = model_dis(pred_t)

        loss_dis_target = bce_logit_loss(dis_output_t,
                                         Variable(torch.FloatTensor(dis_output_t.data.size()).fill_(target_label)).cuda(GPU))
        loss_dis_target.backward()
        loss_dis_value += loss_dis_target.data.cpu().numpy()

        optimizer_dis.step()

        # train S and R together

        # don't accumulate grads in discriminator
        for param in model_dis.parameters():
            param.requires_grad = False

        # train with source
        pred_s = model_seg(image_s)

        if label_exists_t == 1:
            pred_t = model_seg(image_t)
            label_t = Variable(label_t).cuda(GPU)
            loss_seg = dice_coef_loss(pred_s, label_s) + dice_coef_loss(pred_t, label_t)
        else:
            loss_seg = dice_coef_loss(pred_s, label_s)

        loss_seg.backward()
        loss_seg_value += loss_seg.data.cpu().numpy()

        # train with target
        pred_t = model_seg(image_t)
        output_dis = model_dis(pred_t)

        loss_adv_target = LAMBDA_ADV * bce_logit_loss(output_dis,
                                                      Variable(torch.FloatTensor(output_dis.data.size()).fill_(source_label)).cuda(GPU))
        loss_adv_target.backward(retain_graph=True)
        loss_adv_target_value += loss_adv_target.data.cpu().numpy()

        reconstructed_t = model_decoder(pred_t)
        loss_recons = LAMBDA_RECONS * reconstruction_loss(reconstructed_t, image_t)
        loss_recons.backward()

        optimizer_seg.step()
        optimizer_decoder.step()

        print('iter = {0:6d}/{1:6d}, loss_seg = {2:.3f}, loss_recons = {3:.3f}, loss_adv = {4:.3f}, loss_dis = {5:.3f}'.format(
                iter, NUM_STEPS, loss_seg_value, loss_recons_value, loss_adv_target_value, loss_dis_value))

        if iter % SAVE_MODEL_EVERY == 0:
            print('saving model ...')
            torch.save(model_seg.state_dict(), os.path.join(SAVED_MODEL_DIR, 'model_' + str(iter) + '.pth'))

            print('validating...')
            val_adv_target_loss, val_rec_loss, val_iou_score, val_dice_score = validate(model_seg, model_dis,
                                                                                        model_decoder,
                                                                                        target_val_loader,
                                                                                        bce_logit_loss,
                                                                                        reconstruction_loss)

            print('val_iou_score: {0:.6f}, val_dice_score: {1:.6f}'
                  .format(val_iou_score, val_dice_score))

            # best model saving
            if (best_val_iou_score is None) or (val_iou_score > best_val_iou_score):
                print('saving best model so far...')
                torch.save(model_seg.state_dict(),
                           os.path.join(SAVED_MODEL_DIR, 'best_model_' + str(iter) + '.pth'))
                best_val_iou_score = val_iou_score

            model_seg.train()
            model_dis.train()
            model_decoder.train()


if __name__ == '__main__':
    train()
