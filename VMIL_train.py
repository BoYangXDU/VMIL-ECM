"""
@Author: Bo Yang
@Organization: School of Artificial Intelligence, Xidian University
@Email: bond.yang@outlook.com
@LastEditTime: 2024-07-31
"""


def train_epoch(epoch, data_loader, model, opt_E, opt_M , E_step, stage, opt):
    print("#" * 20)
    print("epoch: {}".format(epoch))

    model = model.train()
    device = model.device
    for param in model.parameters():
        param.requires_grad = False

    if E_step:
        print("E step")
        for param in model.trans_dim.parameters():
            param.requires_grad = True
        for param in model.encoder_layer.parameters():
            param.requires_grad = True
        for param in model.transformer_encoder.parameters():
            param.requires_grad = True
        for param in model.bag_classifier.parameters():
            param.requires_grad = True
        optimizer = opt_E
    else:
        print("M step")
        for param in model.fe_1dcnn.parameters():
            param.requires_grad = True
        for param in model.fe_fc.parameters():
            param.requires_grad = True
        for param in model.inst_classifier.parameters():
            param.requires_grad = True
        optimizer = opt_M

    epoch_loss = 0

    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        data = batch[0].unsqueeze(1).float().to(device)
        label = batch[1].float().to(device).view(-1, 1)
        if E_step:
            loss, key, cls = model.forward_E(data, label, stage)
        else:
            loss, key, cls = model.forward_M(data, label, stage)
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    print("epoch loss: {}".format(epoch_loss))



