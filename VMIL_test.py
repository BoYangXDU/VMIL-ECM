"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""


import torch
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from utils.cal_nauc_metric import cal_nauc
from torch.utils.data.dataloader import DataLoader
from VMIL_model import generate_model
from VMIL_opts import parse_opts
import gen_simulated_exp as gendata


def test_epoch(model, test_data, device):
    model.eval()
    pred_labels = []
    GT_labels = []

    val_dataloader = DataLoader(
        test_data,
        batch_size=1280,
        shuffle=False,
        num_workers=0,
    )
    for i, batch in enumerate(val_dataloader):
        data = batch[0].unsqueeze(1).float().to(device).view(-1, 1, 1, 211)
        labels = batch[1].float().to(device).view(-1, 1)
        labels = labels.cpu().data.numpy().tolist()
        _, inst_conf = model.forward_body(data)
        tmp_label = inst_conf.cpu().data.numpy().tolist()
        pred_labels.extend(tmp_label)
        GT_labels.extend(labels)

    fpr, tpr, thresholds = roc_curve(GT_labels, pred_labels, pos_label=1)
    roc_auc = auc(fpr, tpr)
    nauc = cal_nauc(fpr, tpr)
    return roc_auc, nauc


def test_optimal(model, test_data, device, checkpoints_path):
    checkpoints = torch.load(checkpoints_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    pred_labels = []
    GT_labels = []

    val_dataloader = DataLoader(
        test_data,
        batch_size=1280,
        shuffle=False,
        num_workers=0,
    )
    for i, batch in enumerate(val_dataloader):
        data = batch[0].unsqueeze(1).float().to(device).view(-1, 1, 1, 211)
        labels = batch[1].float().to(device).view(-1, 1)
        labels = labels.cpu().data.numpy().tolist()
        _, inst_conf = model.forward_body(data)
        tmp_label = inst_conf.cpu().data.numpy().tolist()
        pred_labels.extend(tmp_label)
        GT_labels.extend(labels)

    fpr, tpr, thresholds = roc_curve(GT_labels, pred_labels, pos_label=1)
    roc_auc = auc(fpr, tpr)
    nauc = cal_nauc(fpr, tpr)

    df = pd.DataFrame()
    df['fpr'] = fpr
    df['tpr'] = tpr
    df.to_csv("simulated-results/" + opt.target_type + "/fpr-tpr-" + str(nauc)[:7] + '.csv',
              index=False)

    return roc_auc, nauc
