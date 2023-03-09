# The baseline implementation of resnet, alexnet, vit, and resnet_attention

import argparse
import os
import sys
import datetime
import time
import math
from pathlib import Path
from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from scipy.linalg import hadamard
import cv2

import torch.nn.functional as F

from models_base import alexnet,resnet,resnet_attribute
from utils.dataloader import DATASET
from utils.fix_seeds import fix_random_seeds


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['main_vit_tiny',  'main_vit_base'] + torchvision_archs


def get_args_parser():
    parser = argparse.ArgumentParser('MPCCI', add_help=False)

    # Model params

    parser.add_argument('--img_size',default=128)
    parser.add_argument('--batch_size', default=10, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset',default='breastPublic',type=str)   # COVID ot BreakHis
    parser.add_argument('--num_classes',type=int)
    parser.add_argument('--job_id',default=0)
    parser.add_argument('--model',default='resnet',type=str)

    return parser

def main(args):
    fix_random_seeds(args.seed)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    if args.dataset=='breast':
        args.num_classes=2
        train_dir = './data/breast_cancer/train.txt'
        test_dir = './data/breast_cancer/test.txt'
        val_dir = './data/breast_cancer/val.txt'
    elif args.dataset=='breastPublic':
        args.num_classes = 2
        train_dir = './data/breastPublic/train.txt'
        test_dir = './data/breastPublic/test.txt'
        val_dir = './data/breastPublic/val.txt'
        if args.device == 'cpu':
            args.root = 'E:/dataset/Dataset_BUSI_with_GT'
        else:
            args.root = '/home/yajiezhang/data/Dataset_BUSI_with_GT'

    if args.model=='vit':
        args.img_size=384
    # elif args.model=='alexnet':
    #     args.img_size=256
    else:
        args.img_size = 128
    print(args)
    dataset_train = DATASET(args.dataset, train_dir, args.root, args.img_size, is_train=True)
    print(f"Train data loaded: there are {len(dataset_train)} images.")

    dataset_val = DATASET(args.dataset, val_dir, args.root, args.img_size,is_train=False)
    print(f"Val data loaded: there are {len(dataset_val)} images.")

    dataset_test = DATASET(args.dataset, test_dir, args.root, args.img_size, is_train=False)
    print(f"Test data loaded: there are {len(dataset_test)} images.")


    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.num_workers, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

    if args.model=='resnet':
        model = resnet.Model(args.num_classes)
    elif args.model=='alexnet':
        model=alexnet.Model(args.num_classes)
    # elif args.model=='vit':
    #     model=vit.Model(num_classes=args.num_classes)
    else:
        model=resnet_attribute.Model(args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    model = model.to(device)  ## 部署在GPU  todo
    num_train=len(dataset_train)
    num_test=len(dataset_test)
    num_val=len(dataset_val)
    train_prec=[]
    test_precd=[]
    max_test_prec=0.0
    max_test_epoch=0
    max_val_prec=0.0
    max_val_epoch=0.0

    max_val_acc = 0.0
    max_val_precision = 0.0
    max_val_recall = 0.0
    max_val_TNR = 0.0

    max_test_acc = 0.0
    max_test_precision = 0.0
    max_test_recall = 0.0
    max_test_TNR = 0.0




    for epoch in range(args.epoch):
        model.train()
        loss = 0.0
        prec = 0.0
        for i, (index, img, label,_) in enumerate(data_loader_train):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            raw_logits=model(img)

            raw_loss = criterion(raw_logits, label)

            total_loss = raw_loss
            loss = loss + total_loss
            total_loss.backward()

            optimizer.step()

            pred = raw_logits.max(1, keepdim=True)[1]
            prec += pred.eq(label.view_as(pred)).sum().item()

        print('[EPOCH:{}],loss:{:.4f},prec:{:.4f}'.format(epoch, loss, prec / num_train))
        train_prec.append(prec / num_train)

        if epoch%1==0:
            model.eval()
            val_label_bank=torch.zeros(num_val,args.num_classes).to(device)
            val_GT_bank=torch.zeros(num_val).to(device).long()

            test_label_bank = torch.zeros(num_test,args.num_classes).to(device)
            test_GT_bank = torch.zeros(num_test).to(device).long()


            with torch.no_grad():

                val_loss = 0.0
                val_prec = 0.0
                for i, (index, img, label,_) in enumerate(data_loader_val):
                    img = img.to(device)
                    label = label.to(device)

                    raw_logits = model(img)
                    raw_loss = criterion(raw_logits, label)
                    total_loss = raw_loss
                    val_loss = val_loss + total_loss


                    pred = raw_logits.max(1, keepdim=True)[1]
                    val_prec += pred.eq(label.view_as(pred)).sum().item()
                    val_label_bank[index]=raw_logits
                    val_GT_bank[index]=label

                val_acc, val_precision, val_recall, val_TNR=eval_tfpn(val_label_bank,val_GT_bank,args)
                if val_prec / num_val >= max_val_prec:
                    max_val_prec = val_prec / num_val

                if val_acc>=max_val_acc:
                    max_val_acc=val_acc
                    max_val_epoch = epoch
                    torch.save({
                        'iteration': epoch,
                        'model_state_dict': model.state_dict(),
                    }, os.path.join('{}_resnet18_{}.t'.format(args.dataset, epoch)))
                if val_precision>=max_val_precision:
                    max_val_precision=val_precision
                if val_recall>=max_val_recall:
                    max_val_recall=val_recall
                if val_TNR>=max_val_TNR:
                    max_val_TNR=val_TNR
                # print('epoch:{},loss:{:.4f},prec:{:.4f},max_val_epoch:{},max_val_pre:{:.4f}'.format(epoch, val_loss,
                #                                                                                  val_prec / num_val,
                #                                                                                  max_val_epoch,
                #                                                                                  max_val_prec))
                # print()
                print('VAL: [EPOCH/max:{}/{}], acc/max:{:.4f}/{:.4f},precision/max:{:.4f}/{:.4f},recall/max:{:.4f}/{:.4f},TRN/max:{:.4f}/{:.4f}'.format(
                    epoch,max_val_epoch,val_acc,max_val_acc,val_precision,max_val_precision,val_recall,max_val_recall,val_TNR,max_val_TNR
                ))

                test_loss = 0.0
                test_prec = 0.0
                for i, (index, img, label,_) in enumerate(data_loader_test):
                    img = img.to(device)
                    label = label.to(device)

                    raw_logits = model(img)
                    raw_loss = criterion(raw_logits, label)
                    total_loss = raw_loss
                    test_loss = test_loss + total_loss

                    pred = raw_logits.max(1, keepdim=True)[1]
                    test_prec += pred.eq(label.view_as(pred)).sum().item()

                    test_label_bank[index]=raw_logits
                    test_GT_bank[index]=label

                test_acc, test_precision, test_recall, test_TNR = eval_tfpn(test_label_bank, test_GT_bank, args)

                if test_prec/num_test>=max_test_prec:
                    max_test_prec=test_prec/num_test

                if test_acc>=max_test_acc:
                    max_test_acc=test_acc
                    max_test_epoch = epoch
                if test_precision>=max_test_precision:
                    max_test_precision=test_precision
                if test_recall>=max_test_recall:
                    max_test_recall=test_recall
                if test_TNR>=max_test_TNR:
                    max_test_TNR=test_TNR
                # print('epoch:{},loss:{:.4f},prec:{:.4f},max_test_epoch:{},max_test_pre:{:.4f}'.format(epoch, test_loss, test_prec / num_test,max_test_epoch,max_test_prec))
                # test_precd.append(test_prec / num_test)
                # print()
                print(
                    'TEST:[EPOCH/max:{}/{}], acc/max:{:.4f}/{:.4f},precision/max:{:.4f}/{:.4f},recall/max:{:.4f}/{:.4f},TRN/max:{:.4f}/{:.4f}'.format(
                        epoch,max_test_epoch,test_acc, max_test_acc, test_precision, max_test_precision, test_recall, max_test_recall, test_TNR,
                        max_test_TNR
                    ))
    # torch.save(train_prec,'./train_prec_ce_aa_ra_'+args.model)
    # torch.save(test_precd, './test_prec_ce_aa_ra_' + args.model)

def eval_tfpn(output, target, args):
    # print((torch.stack(output)).shape)
    # output=torch.stack(output).view(-1,num_classes)
    # target=torch.stack(target).view(-1)
    # target= torch.tensor(np.eye(num_classes, dtype=np.uint8)[target.cpu().numpy()]).to(
    #     device).float()
    output=torch.softmax(output,dim=1)
    output=torch.max(output,dim=1)[1]
    if args.dataset=='breastPublic':
        output=1-output
        target=1-target
    # print(output)
    # print(target)
    output=output.cpu().numpy()
    target=target.cpu().numpy()


    # output = output > threshold
    # target = target > 0.5

    tp = np.sum(output*target)
    fp = np.sum(output*(1-target))
    fn = np.sum((1-output)*target)
    tn = np.sum((1-output)*(1-target))

    # print(tp,fp,fn,tn)

    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 =2*precision*recall/(precision+recall)

    return acc, precision, recall, F1

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # print(args)
    start_train = time.time()
    main(args)
    end_train = time.time()
    print('Training time in: %s' %((end_train - start_train)/3600))
