# Secondary development based on ConvNeXt.git:
# https://github.com/facebookresearch/ConvNeXt.git


import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from torchvision.transforms import InterpolationMode

import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'

# for model create
from models import *

import warnings

warnings.filterwarnings("ignore")

from configs import digsfunc


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)

    # multimodal data
    parser.add_argument('--multimodal', default=False,
                        help='multimodal data adding control')
    parser.add_argument('--multi-func', default=digsfunc,
                        help='multimodal data adding interface')

    # Model parameters
    parser.add_argument('--model', default='efficientnet_b0', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='pretrain weights to transfer learning')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # * Finetuning params
    parser.add_argument('--finetune', default='output/rocks_extend/efficientnet_b0/checkpoint-best.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/DATA/DATA/lzw/data/6C_train_split_extend/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')

    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='output/rocks_extend/googlenet/checkpoint-best.pth',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)

    parser.add_argument('--eval', type=str2bool, default=True,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def _get(args):
    # utils.init_distributed_mode(args)
    # print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    model = create_model(
        args.model,
        pretrained=args.pretrain,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    return model


def get(ckpt=None):
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if ckpt:
        args.finetune = ckpt
    model = _get(args)
    return model


if __name__ == '__main__':
    model = get()
    data = torch.rand(1, 3, 224, 224).cuda()
    out = model(data)
    print(out.shape)
