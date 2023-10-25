import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from model.trans_forcer import TransForcer
from model.resnet_fcn import resnet_fcn
from data.data import ForceData
from functools import reduce
import tensorboardX

def criterion(pred, gt):
    loss = nn.CrossEntropyLoss()
    losses = []
    for y, yt in zip(pred, gt):
        losses.append(loss(y, yt))
    loss_sum = reduce(lambda x,y:x+y, losses)

    return losses, loss_sum

def train(net, criterion, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, stats_dir, optim='sgd', init=True, scheduler_type='Cosine'):
    def init_xavier(m): #参数初始化
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)
    
    if optim == 'sgd': 
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    tb = tensorboardX.SummaryWriter(stats_dir)

    lowest_loss = 100000
    for epoch in range(num_epoch):

        print("——————{} epoch——————".format(epoch + 1))

        result = {"loss": 0, "loss_x": 0, "loss_y": 0, "loss_z": 0}
        net.train()
        for batch in tqdm(train_dataloader, desc='training'):
            rgb, inf, targets, _ = batch
            rgb = rgb.to(device)
            inf = inf.to(device)
            labels = [target.to(device) for target in targets]
            output = net(rgb, inf)

            loss_list, Loss = criterion(output, labels)
          
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            result['loss'] += Loss.item() / len(train_dataloader)
            result['loss_x'] += loss_list[0].item() / len(train_dataloader)
            result['loss_y'] += loss_list[1].item() / len(train_dataloader)
            result['loss_z'] += loss_list[2].item() / len(train_dataloader)

        scheduler.step()

        tb.add_scalar('loss/sum_loss', result['loss'], epoch)
        tb.add_scalar('loss/x_loss', result['loss_x'], epoch)
        tb.add_scalar('loss/y_loss', result['loss_y'], epoch)
        tb.add_scalar('loss/z_loss', result['loss_z'], epoch)
        print("epoch: {0}, Loss: {1}, LossX: {2}, LossY: {3}, LossZ{4}".format(epoch, result['loss'],
            result['loss_x'], result['loss_y'], result['loss_z']))

        net.eval()
        eval_loss = 0
        with torch.no_grad():
            for rgb, inf, targets, _ in valid_dataloader:
                rgb = rgb.to(device)
                inf = inf.to(device)
                labels = [target.to(device) for target in targets]
                output = net(rgb, inf)
                loss_list, Loss = criterion(output, labels)
                eval_loss += Loss
        
        eval_losses = eval_loss / (len(valid_dataloader))
        if eval_losses < lowest_loss:
            lowest_loss = eval_losses
            torch.save(net.state_dict(), os.path.join(stats_dir, "best.pth"))
        tb.add_scalar('loss/eval_loss', eval_losses, epoch)
        print("val Loss: {}".format(eval_losses))

def split_dataset(dataset, split, batch_size, num_workers):
    indices = list(range(len(dataset)))
    split = int(np.floor(split * len(dataset)))
    np.random.seed(1234)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )
    return train_data, val_data

if __name__ == "__main__":
    stats_dir = "stats_10_5_1"
    data_path = "src/data2"
    checkout_path = "/home/zjw/learn/cv_self/force_map/backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
    batch_size = 32
    img_shape = (256, 256)
    epoch = 100
    lr = 0.001
    lr_min = 0.00001
    num_workers = 12
    decoder_params = {"depth": 4, "hidden_dim": 96, "norm_type": dict(type="BN"), "act_type": dict(type="LeakyReLU")}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ForceData(data_path)

    train_dataloader, val_dataloader = split_dataset(dataset, 0.9, batch_size, num_workers, )

    net = TransForcer(in_channels=6, window_size=8, input_shape=img_shape, checkout_path=checkout_path, use_checkout=False, **decoder_params)
    train(net, criterion, train_dataloader, val_dataloader, device, batch_size, epoch, lr, lr_min, stats_dir)
