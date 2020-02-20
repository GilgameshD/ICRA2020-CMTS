'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-02-20 13:53:49
@LastEditTime: 2020-02-20 14:00:45
@Description: 
'''

import argparse
import torch
import os

from trainer import Trainer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDT')

    parser.add_argument('-continue_training', default=False, type=bool, help='continue training or not')
    parser.add_argument('-type', default=0, type=int, help='type of stage, 0 means train, 1 means test')
    parser.add_argument('-id', default=1, type=int, help='index of the model')
    parser.add_argument('-max_epoch', default=200, type=int, help='maximum training iteration')
    parser.add_argument('-gpu', default=0, type=int, help='choose gpu number')
    parser.add_argument('-batch_size', default=512, type=int, help='batch size')
    parser.add_argument('-print_iter', default=10, type=int, help='print losses iter')
    parser.add_argument('-save_epoch', default=1, type=int, help='the ieration that save models')
    parser.add_argument('-num_sample', default=10, type=int, help='the number of the samples')

    parser.add_argument('-image_size', default=128, type=int, help='size of source and target domain image')
    parser.add_argument('-z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate of the model')
    parser.add_argument('-alpha', default=1, type=float, help='weight of reconstruction error')
    parser.add_argument('-beta', default=0.5, type=float, help='weight of kl divergence')
    parser.add_argument('-gamma', default=0.1, type=float, help='weight of fusion loss')

    parser.add_argument('-sample_path', default='./samples', type=str, help='output directory')
    parser.add_argument('-model_path', default='./model', type=str, help='checkpoint directory')
    parser.add_argument('-extended_path', default='./extended', type=str, help='checkpoint directory')

    parser.add_argument('-normal_data_path', default='./argoverse-data/data.normal.train.npy', type=str, help='dataset directory')
    parser.add_argument('-normal_map_path', default='./argoverse-data/map.normal.train.npy', type=str, help='dataset directory')
    parser.add_argument('-collision_data_path', default='./argoverse-data/data.collision.train.npy', type=str, help='dataset directory')
    parser.add_argument('-collision_map_path', default='./argoverse-data/map.collision.train.npy', type=str, help='dataset directory')

    parser.add_argument('-num_workers', default=8, type=int, help='dataloader num_workers')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    net = Trainer(args)

    if args.type == 0:
        net.train()
    elif args.type == 1:
        net.sample()
    elif args.type == 2:
        net.fusion()
    elif args.type == 3:
        net.generate()
    elif args.type == 4:
        net.encode()
    else:
        print('No match type')
