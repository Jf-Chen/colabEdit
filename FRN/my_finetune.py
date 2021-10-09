# 要带着GPU才行，不然会出错
# 下载weight之后的fine-tune部分
# 5-way 5-shot执行大约5小时
import os
import sys
import torch
import yaml
import argparse
from functools import partial

from datasets import dataloaders
from models.FRN import FRN
from trainers import trainer, frn_train
from utils import util

parser = argparse.ArgumentParser()

## general hyper-parameters
parser.add_argument("--opt",help="optimizer",choices=['adam','sgd'])
parser.add_argument("--lr",help="initial learning rate",type=float)
parser.add_argument("--gamma",help="learning rate cut scalar",type=float,default=0.1)
parser.add_argument("--epoch",help="number of epochs before lr is cut by gamma",type=int)
parser.add_argument("--stage",help="number lr stages",type=int)
parser.add_argument("--weight_decay",help="weight decay for optimizer",type=float)
parser.add_argument("--gpu",help="gpu device",type=int,default=0)
parser.add_argument("--seed",help="random seed",type=int,default=42)
parser.add_argument("--val_epoch",help="number of epochs before eval on val",type=int,default=20)
parser.add_argument("--resnet", help="whether use resnet12 as backbone or not",action="store_true")
parser.add_argument("--nesterov",help="nesterov for sgd",action="store_true")
parser.add_argument("--batch_size",help="batch size used during pre-training",type=int)
parser.add_argument('--decay_epoch',nargs='+',help='epochs that cut lr',type=int)
parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test",action="store_true")
parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep",action="store_true")
parser.add_argument("--train_way",help="training way",type=int)
parser.add_argument("--test_way",help="test way",type=int,default=5)
parser.add_argument("--train_shot",help="number of support images per class for meta-training and meta-testing during validation",type=int)
parser.add_argument("--test_shot",nargs='+',help="number of support images per class for meta-testing during final test",type=int)
parser.add_argument("--train_query_shot",help="number of query images per class during meta-training",type=int,default=15)
parser.add_argument("--test_query_shot",help="number of query images per class during meta-testing",type=int,default=16)
parser.add_argument("--train_transform_type",help="size transformation type during training",type=int)
parser.add_argument("--test_transform_type",help="size transformation type during inference",type=int)
parser.add_argument("--val_trial",help="number of meta-testing episodes during validation",type=int,default=1000)
parser.add_argument("--detailed_name", help="whether include training details in the name",action="store_true")


args,unkonwn = parser.parse_known_args()
# args,unkonwn = parser.parse_known_args('--opt sgd --lr 1e-3 --gamma 1e-1 --epoch 150 --decay_epoch 70 120 --val_epoch 5 --weight_decay 5e-4 --nesterov --train_transform_type 0 --resnet --train_shot 5 --train_way 20 --test_shot 1 5 --pre --gpu 0'.split() )


args.opt='sgd'
args.lr=1e-1
args.gamma=1e-1
args.epoch=150
args.decay_epoch=70,120


args.val_epoch=5
args.weight_decay=5e-4
args.nesterov=True
args.train_transform_type=0

args.resnet=True
args.train_shot=5 # 
args.train_way=5
args.train_way=5 # 原本是20，但显存不够用
args.test_shot=1,5
args.pre=True
args.gpu=0







with open('config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'mini-ImageNet')


pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = FRN(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)

pretrained_model_path = '/content/colabEdit/FRN/trained_model_weights/mini-ImageNet/FRN/ResNet-12_pretrain/model.pth'
# pretrained_model_path = '../ResNet-12_pretrain/model_ResNet-12.pth'
#pretrained_model_path = '../../../../trained_model_weights/mini-ImageNet/FRN/ResNet-12_pretrain/model.pth'

model.load_state_dict(torch.load(pretrained_model_path,map_location=util.get_device_map(args.gpu)),strict=False)

train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)
