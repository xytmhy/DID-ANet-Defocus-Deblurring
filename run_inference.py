import argparse
import os
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import deblur_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from matplotlib import pyplot

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch Deblur inference on a folder of images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default='/media/data/test/',
                    help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', default='./model_best.pth.tar',
                    help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'ResultDeblur'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading code
    input_transform = transforms.Compose([
        deblur_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    imgs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*.{}'.format(ext))
        for file in test_files:
            imgs.append(file)

    print('{} samples found'.format(len(imgs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    for img_file in tqdm(imgs):

        img = imread(img_file)
        img = img[:, :, :3]
        img = input_transform(img)
        input_var = img.unsqueeze(0)

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)

        for concat_output in output:
            result_rgb = tensor2rgb(concat_output)

            result_deblur = result_rgb[1:, :].clip(0, 1)
            deblur_save = (result_deblur * 255).astype(np.uint8).transpose(1, 2, 0)
            pyplot.imsave(save_path/'{}{}.png'.format(img_file.namebase, 'deblur'), deblur_save)

            result_defest = result_rgb[:1, :].clip(0, 1)
            defest_save = (result_defest * 255).astype(np.uint8).transpose(1, 2, 0)
            defest_save = np.squeeze(defest_save)
            pyplot.imsave(save_path / '{}{}.png'.format(img_file.namebase, 'defest'), defest_save, cmap=pyplot.cm.gray, vmin=0, vmax=255)


def tensor2rgb(img_tensor):
    map_np = img_tensor.detach().cpu().numpy()
    _, h, w = map_np.shape

    return map_np


if __name__ == '__main__':
    main()
