"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""

import torch.optim as optim
import os
import heapq
import torch
from torch.utils.data.dataloader import DataLoader
from VMIL_model import generate_model
from VMIL_train import train_epoch
from VMIL_test import test_epoch
from VMIL_opts import parse_opts
import gen_simulated_exp as gendata
algorithem = 'VMIL-ECM'


def main():
    print('')
    print("training EM model")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt = parse_opts()
    torch.manual_seed(opt.manual_seed)
    device = torch.device("cuda")
    model, parameters = generate_model(opt)

    # E_step
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder_layer.parameters():
        param.requires_grad = True
    for param in model.transformer_encoder.parameters():
        param.requires_grad = True
    for param in model.bag_classifier.parameters():
        param.requires_grad = True
    opt_E = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.learning_rate)

    # M_step
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fe_1dcnn.parameters():
        param.requires_grad = True
    for param in model.fe_fc.parameters():
        param.requires_grad = True
    for param in model.inst_classifier.parameters():
        param.requires_grad = True
    opt_M = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=opt.learning_rate)

    train_datapath = 'xxxxx.mat'
    val_datapath ='xxxxx.mat'
    test_datapath = 'xxxxx.mat'

    train_dataset = gendata.train_data_cause(train_datapath)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
    )
    data_test = gendata.tes_data_cause(test_datapath)
    data_val  = gendata.val_data_cause(val_datapath)
    max_auc = 0.0
    E_step = False

    auc_list =[]
    for epoch in range(1, opt.t_epoches):

        if epoch <= opt.p_epoches:
            stage = 1
            if epoch % 2 == 0:
                E_step = not E_step

        elif epoch > opt.p_epoches:
            stage = 2
            if epoch % 5 == 0:
                E_step = not E_step

        train_epoch(epoch, train_dataloader, model, opt_E, opt_M, E_step, stage, opt)
        auc, nauc = test_epoch(model, data_val, device)
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        if epoch > opt.p_epoches:
            auc_list.append(auc)
        if auc >= max_auc:
            max_auc = auc
            torch.save(save_dict, 'simulated-results/' + opt.target_type + '/auc-' + str(format(auc, '.5f')) + '-nauc-' +  str(format(nauc, '.5f')) + '.pt')

        print("current auc: {} and nauc: {}; all time max auc: {}".format(
            round(auc, 5), round(nauc, 5), round(max_auc, 5)))

    max_number = heapq.nlargest(20, auc_list)
    max_index = []
    for t in max_number:
        index = auc_list.index(t)
        max_index.append(index+opt.p_epoches)
        auc_list[index] = 0

    print('after warm-up: max auc', max_number)
    print('ater warm-up: max auc index', max_index)

    return

