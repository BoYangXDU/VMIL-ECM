"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""

import torch
import torch.nn as nn


def generate_model(opt):
    device = torch.device("cuda")
    model = VMIL(device=device,opt=opt).to(device)
    torch.backends.cudnn.enabled = False
    return model, model.parameters()


class VMIL(nn.Module):

    def __init__(self, device, opt):

        super(VMIL, self).__init__()
        self.L = 256
        self._nlayer = opt._nlayer
        self._gamma = opt._gamma
        self.BCE = torch.nn.BCELoss(reduction='sum')
        self.KL = torch.nn.KLDivLoss(reduction='sum')
        self.device = device

        self.fe_1dcnn = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(20, 128, kernel_size=(1, 3), ),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(128, 64, kernel_size=(1, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.fe_fc = nn.Sequential(
            nn.Linear(opt._fc_dim, self.L),
            nn.ReLU(),
        )

        self.trans_dim = nn.Sequential(
            nn.Linear(opt._spectral_dim, self.L),
            nn.ReLU(),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.L,
            nhead=8,
            dim_feedforward=64,
            dropout=0,
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self._nlayer)

        self.bag_classifier = nn.Sequential(
            nn.Linear(self.L, 1),
            nn.Sigmoid()
        )

        self.inst_classifier = nn.Sequential(
            nn.Linear(self.L, 1),
            nn.Sigmoid()
        )

    def forward_body(self, hsi_input):
        H = self.fe_1dcnn(hsi_input).transpose(1, 2)
        H = H.reshape(H.shape[0], H.shape[1], H.shape[2] * H.shape[3])
        H = self.fe_fc(H)
        cls = self.inst_classifier(H).squeeze(2)  #

        H_tf = self.transformer_encoder(self.trans_dim(hsi_input.squeeze(1)))
        key = self.bag_classifier(H_tf).squeeze(2) # q
        return key, cls

    def forward_E(self, hsi_input, label, stage):
        spread_labels = label.repeat(1, hsi_input.shape[2])
        key, cls = self.forward_body(hsi_input)
        loss = 0

        cls = cls.clone().detach()  # p
        cls[spread_labels == 0] = 0
        loss += self.BCE(key, cls)

        return loss, key, cls

    def forward_M(self, hsi_input, label, stage):
        key, cls = self.forward_body(hsi_input)
        loss = 0

        if stage == 1:
            spread_labels = label.repeat(1, hsi_input.shape[2])
            loss += self.BCE(cls, spread_labels)
        elif stage == 2:
            spread_labels = label.repeat(1, hsi_input.shape[2])
            psuodo_labels = torch.zeros(spread_labels.shape).to(self.device)
            key = key.clone().detach()

            threshold = torch.min(key, dim=1)[0] + self._gamma * \
                        (torch.max(key, dim=1)[0] - torch.min(key, dim=1)[0])
            threshold = threshold.unsqueeze(1).repeat(1, hsi_input.shape[2])

            psuodo_labels[key > threshold] = 1
            psuodo_labels[spread_labels == 0] = 0

            loss += self.BCE(cls, psuodo_labels)
        return loss, key, cls








