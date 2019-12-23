'''
@Author: Tao Hang
@Date: 2019-10-11 15:05:59
@LastEditors  : Tao Hang
@LastEditTime : 2019-12-19 02:19:06
@Description: 
'''
import os
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image


class TextLineDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_root,
                 text_line_file=None,
                 transform=None,
                 target_transform=None):
        self.image_root = image_root
        self.text_line_file = text_line_file
        with open(text_line_file, 'r', encoding='utf8') as fp:
            self.lines = fp.readlines()
            self.num_samples = len(self.lines)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        line_splits = self.lines[index].strip().split(' ')
        image_path = os.path.join(self.image_root, line_splits[0])
        # print()
        try:
            img = Image.open(image_path).convert('RGB')
            # print(img.data)
        except IOError:
            print('Corrupted image for {}'.format(index))
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label = torch.IntTensor(list(map(int, line_splits[1:])))

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class ResizeNormalize(object):
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img: Image):
        img = np.array(img)
        h, w, c = img.shape
        height = self.img_height
        width = int(w * height / h)
        if width >= self.img_width:
            img = cv2.resize(img, (self.img_width, self.img_height))
        else:
            img = cv2.resize(img, (width, height))
            img_pad = np.zeros((self.img_height, self.img_width, c),
                               dtype=img.dtype)
            img_pad[:height, :width, :] = img
            img = img_pad

        # print(img[:, :, 0])
        img = Image.fromarray(img)

        # img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        img = self.toTensor(img)
        # print(img.data)
        img.sub_(0.5).div_(0.5)

        return img


class RandomSequentialSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data, batch_size):
        self.num_samples = len(data)
        self.batch_size = batch_size

    def __iter__(self):
        n_batches = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batches):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index

        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):
    def __init__(self, img_height=32, img_width=280):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = ResizeNormalize(img_height, img_width)

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([image.unsqueeze(0) for image in images], 0)

        return images, labels
