'''
@Author: Tao Hang
@Date: 2019-12-23 09:48:21
@LastEditors  : Tao Hang
@LastEditTime : 2019-12-23 15:01:24
@Description: 
'''
import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

import net.crnn_ctc as crnn_ctc
import src.convert as convert
import src.dataset as dataset
import src.utils as utils


def eval(net, device, cfg, input, converter):
    net = net.to(device)

    net.eval()

    output = net(input)

    preds = output.max(2)[1]
    preds = preds.permute(1, 0)


def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alphabet = utils.get_alphabet('./data/char_std_5990.txt')
    num_classes = len(alphabet)
    converter = convert.StringLabelConverter(alphabet)
    transformer = dataset.ResizeNormalize(32, 280)

    model = crnn_ctc.CRNN(
        in_channels=3,
        hidden_size=256,
        output_size=num_classes,
    )
    net = model.to(device)

    if os.path.exists(cfg.model_path):
        print('Loading model from {0}'.format(cfg.model_path))
        model.load_state_dict(torch.load(cfg.model_path))
        print('Done!')

    if os.path.exists(cfg.image_path):
        image = Image.open(cfg.image_path).convert('RGB')
        image = transformer(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(device)

    net.eval()
    output = net(image)
    preds = output.max(2)[1]
    # print(preds)
    preds_len = torch.IntTensor([preds.size(0)] * int(preds.size(1)))
    results = converter.decode(preds, preds_len)
    print('Result: {0}'.format(results))
    # with open(cfg.save_path, 'w', encoding='utf8') as fp:
    #     fp.write(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        default='./20190926165839.png',
        help='path of test image.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./model/crnn_ctc_4.pth',
        help='path of model.',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./output.txt',
        help='path of results',
    )
    args = parser.parse_args()
    main(args)
