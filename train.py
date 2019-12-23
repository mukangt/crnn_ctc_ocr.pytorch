'''
@Author: Tao Hang
@Date: 2019-10-17 16:52:25
@LastEditors  : Tao Hang
@LastEditTime : 2019-12-23 11:59:15
@Description: 
'''

import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision

import net.crnn_ctc as crnn_ctc
import src.convert as convert
import src.dataset as dataset
import src.utils as utils


def matrix2linear(labels):
    labels_res = torch.IntTensor([])
    labels_len = []
    for label in labels:
        labels_len.append(len(label))
        labels_res = torch.cat((labels_res, label), 0)
    return labels_res, labels_len


def decode(preds):
    pred = []
    # print(preds)
    for i in range(len(preds)):
        if preds[i] != 0 and ((i == 0) or
                              (i != 0 and preds[i] != preds[i - 1])):
            pred.append(int(preds[i]))
    return pred


def valid(net, valid_loader, device, cfg):
    print('start valid')
    criterion = nn.CTCLoss()

    loss_avg = utils.Averager()

    net.eval()
    # net = net.to(device)
    correct_num = 0
    total_num = 0

    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels, labels_len = matrix2linear(labels)
        labels = labels.to(device)
        labels_len = torch.IntTensor(labels_len)

        preds = net(images)
        # if torch.sum(torch.isnan(preds)) >= 1:
        #     print('nan: {}, lr: {}'.format(i + 1, scheduler.get_lr()[0]))
        #     break

        preds_len = torch.IntTensor([preds.size(0)] * int(preds.size(1)))
        with torch.backends.cudnn.flags(enabled=False):
            loss = criterion(preds, labels, preds_len, labels_len)
        loss_avg.add(loss)
        preds = preds.max(2)[1]
        # print(preds.size())
        preds = preds.transpose(1, 0).contiguous().view(-1)
        # print(preds.size())
        preds = decode(preds)
        # print(len(preds))
        total_num += len(preds)
        for x, y in zip(preds, labels):
            if int(x) == int(y):
                correct_num += 1
    acc = correct_num / float(total_num) * 100
    valid_loss = loss_avg.val()
    print('Valid Loss: {0:.3f}, Accuracy: {1:.3f}%'.format(valid_loss, acc))


def train(net, train_loader, valid_loader, device, cfg):
    criterion = nn.CTCLoss()
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=cfg.learning_rate)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.1,
    )

    loss_avg = utils.Averager()

    net = net.to(device)

    for epoch in range(cfg.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            net.train()
            images = images.to(device)
            labels, labels_len = matrix2linear(labels)
            labels = labels.to(device)
            labels_len = torch.IntTensor(labels_len)

            preds = net(images)
            # print(preds.size(), preds_len)
            if torch.sum(torch.isnan(preds)) >= 1:
                print('nan: {}, lr: {}'.format(i + 1, scheduler.get_lr()[0]))
                break

            preds_len = torch.IntTensor([preds.size(0)] * int(preds.size(1)))
            with torch.backends.cudnn.flags(enabled=False):
                loss = criterion(preds, labels, preds_len, labels_len)

            loss_avg.add(loss)

            if (i + 1) % cfg.display_interval == 0:
                print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4:.3f}'.format(
                    epoch + 1,
                    cfg.num_epochs,
                    i + 1,
                    len(train_loader),
                    loss_avg.val(),
                ))
                loss_avg.reset()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=20,
                norm_type=2,
            )
            optimizer.step()
            if (i + 1) % cfg.valid_interval == 0:
                valid(net, valid_loader, device, cfg)

        scheduler.step()
        torch.save(net.state_dict(),
                   '{0}/crnn_ctc_{1}.pth'.format(cfg.model_path, epoch))


def main(cfg):
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)

    torch.manual_seed(cfg.seed)
    if cfg.gpu_id is not None and torch.cuda.is_available():
        # os.environ['CUDA_VISIBLE_DEVICE'] = cfg.gpu_id
        device = torch.device('cuda:{0}'.format(cfg.gpu_id))
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    else:
        device = torch.device('cpu')

    alphabet = utils.get_alphabet(cfg.alpha_path)
    # converter = convert.StringLabelConverter(alphabet)
    num_classes = len(alphabet)

    # prepare train data
    train_dataset = dataset.TextLineDataset(
        cfg.image_root,
        cfg.train_list,
        transform=dataset.ResizeNormalize(
            img_width=cfg.img_width,
            img_height=cfg.img_height,
        ),
    )
    # sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        # sampler=sampler,
        shuffle=True,
    )

    # prepare test data
    valid_dataset = dataset.TextLineDataset(
        cfg.image_root,
        cfg.valid_list,
        transform=dataset.ResizeNormalize(
            img_width=cfg.img_width,
            img_height=cfg.img_height,
        ),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )

    model = crnn_ctc.CRNN(
        in_channels=3,
        hidden_size=cfg.hidden_size,
        output_size=num_classes,
    )
    if not cfg.pretrained:
        model.apply(utils.weights_init)

    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     model = nn.DataParallel(model)
    model = model.to(device)

    train(model, train_loader, valid_loader, device, cfg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=5000,
        help='interval of display valid information',
    )
    parser.add_argument(
        '--display_interval',
        type=int,
        default=100,
        help='interval of display train information',
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="1",
        help="number of gpus used to train",
    )
    parser.add_argument(
        "--alpha_path",
        type=str,
        default="./data/char_std_5990.txt",
        help="path of alphabet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="seed for cpu and gpu",
    )
    parser.add_argument(
        "--train_list",
        type=str,
        default="/home/hangtao/DataSet/data_train.txt",
        help="path to train dataset list file",
    )
    parser.add_argument(
        "--valid_list",
        type=str,
        default="/home/hangtao/DataSet/data_test.txt",
        help="path to evalation dataset list file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of data loading num_workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size",
    )
    parser.add_argument(
        '--image_root',
        type=str,
        default='/home/hangtao/DataSet/images',
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=32,
        help="the height of the input image to network",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=280,
        help="the width of the input image to network",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="size of the lstm hidden state",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="learning rate for Critic, default=0.00005",
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--model_path",
        default="./model/",
        help="Where to store samples and models",
    )
    parser.add_argument(
        "--random_sample",
        default=True,
        action="store_true",
        help="whether to sample the dataset with random sampler",
    )
    parser.add_argument(
        "--pretrained",
        default=False,
        help="where to use teach forcing",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=71,
        help="the width of the feature map out from cnn",
    )
    args = parser.parse_args()

    main(args)
