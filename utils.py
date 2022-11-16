import matplotlib.pyplot as plt
import os
import numpy as np
import torch


# Function for obtaining the training folder given dropout rate, learning rate and batch size
def obtain_training_folder(dataset_name, dr, lr, bs):
    root = os.path.join('experiments', dataset_name)
    base = os.path.join(root, 'dr{}'.format(str(dr).split('.')[1]))
    path = os.path.join(base, 'lr{:.5}_bs{}'.format(str('{:.5f}'.format(lr)).split('.')[1], bs))

    # Creating the folder if it does not exists
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(base):
        os.mkdir(base)
    if not os.path.exists(path):
        os.mkdir(path)

    return path


# Function for computing the loss on an entire data set (training set by default)
def compute_loss_on_dataloader(m, ds, dl, device, loss_function, mode='training'):
    ds.switch_mode(mode)
    ds_len, loss = dl.__len__(), 0
    for sample_batched in dl:
        X, y = sample_batched['x'].to(device), sample_batched['y'].to(device)
        y = torch.unsqueeze(y, dim=1)
        pred = m(X)
        loss += (torch.sqrt(loss_function(pred, y))).item()
    return loss / ds_len



