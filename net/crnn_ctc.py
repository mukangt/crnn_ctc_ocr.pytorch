'''
@Author: Tao Hang
@Date: 2019-10-11 15:07:38
@LastEditors: Tao Hang
@LastEditTime: 2019-12-17 02:15:56
@Description: 
'''
import torch
import torch.nn as nn
# import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1),
                                  nn.ReLU(True), nn.MaxPool2d(2, 2),
                                  nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(128, 256, 3, 1, 1),
                                  nn.BatchNorm2d(256), nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                                  nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                                  nn.Conv2d(256, 512, 3, 1, 1),
                                  nn.BatchNorm2d(512), nn.ReLU(True),
                                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
                                  nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
                                  nn.Conv2d(512, 512, 2, 1, 0),
                                  nn.BatchNorm2d(512), nn.ReLU(True))

    def forward(self, input):
        output = self.conv(input)
        return output


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, N, C = recurrent.size()
        t_rec = recurrent.view(T * N, C)

        output = self.embedding(t_rec)
        output = output.view(T, N, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, in_channels, hidden_size, output_size):
        super(CRNN, self).__init__()

        self.cnn = CNN(in_channels)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, output_size))

    def forward(self, input):
        conv = self.cnn(input)
        N, C, H, W = conv.size()

        assert H == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)  # [N, C, H, W] -> [N, C, W]
        conv = conv.permute(2, 0, 1)  # [N, C, W] -> [T, N, C]
        output = self.rnn(conv)
        return output