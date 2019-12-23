'''
@Author: Tao Hang
@Date: 2019-10-11 15:05:51
@LastEditors  : Tao Hang
@LastEditTime : 2019-12-18 21:53:06
@Description: 
'''
import collections

import torch
import torch.nn as nn


class StringLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet  # for '-1' index
        # print(len(alphabet))
        self.dict = {}
        # self.blank_ind = len(self.alphabet) - 1
        self.blank_ind = 0
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char.lower()] = i + 1

    def get_blank_ind(self):
        return self.blank_ind

    def encode(self, text):
        '''
        @description: Support batch or single str.
        @param {text: texts to convert} 
        @return: 
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        '''

        if isinstance(text, str):
            label = [self.dict[char.lower()] for char in text]
            length = [len(label)]
        elif isinstance(text, collections.Iterable):
            length = [len(t) for t in text]
            label, _ = self.encode(''.join(text))

        return torch.IntTensor(label), torch.IntTensor(length)

    def decode(self, label, length, raw=False):
        '''
        @description: Decode labels back into strs
        @param {
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts,
            torch.IntTensor [n]: length of each text
            } 
        @return: text (str or list of str): texts to convert
        '''
        if length.numel() == 1:
            length = length[0]
            assert label.numel(
            ) == length, 'label with length: {} does not match declared length: {}'.format(
                label.numel(), length)
            if raw:
                return ''.join([self.alphabet[i] for i in label])
            else:
                char_list = []
                for i, lab in enumerate(label):
                    if lab != self.blank_ind and (
                            not (i > 0 and lab == label[i - 1])):
                        char_list.append(self.alphabet[lab - 1])
                return ''.join(char_list)
        else:
            assert label.numel() == length.sum(
            ), 'label with length: {} does not match declared length: {}'.format(
                label.numel(), length.sum())
            text = []
            index = 0
            for l in length:
                text.append(
                    self.decode(label[index:index + l], torch.IntTensor([l]),
                                raw))
                index += l

            return text
