# my_main.py,测试train.py使用
import os
import sys
import torch
import yaml
import argparse
from functools import partial
#from trainers import trainer, frn_train
#from datasets import dataloaders
#from models.FRN import FRN


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

args = parser.parse_args(args=[])

args.opt='sgd'
args.lr=1e-1
args.gamma=350
args.decay_epoch=200,300

args.batch_size=128
args.val_epoch=25
args.weight_decay=5e-4
args.nesterov=True
args.train_transform_type=0

args.resnet=True
args.train_shot=1
args.test_shot=1,5
args.pre=True
args.gpu=0

with open('config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])# 
fewshot_path = os.path.join(data_path,'mini-ImageNet')

pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)
train_loader = dataloaders.normal_train_dataloader(data_path=pm.train,
                                                batch_size=args.batch_size,
                                                transform_type=args.train_transform_type)
num_cat = len(train_loader.dataset.classes)

model = FRN(is_pretraining=True,
            num_cat=num_cat,
            resnet=args.resnet)

train_func = partial(frn_train.pre_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

print(tm)
"""
print("data_pat",data_path)
print("fewshot_path",fewshot_path)
print("num_cat:",num_cat)
print("model.resolution:",model.resolution)
print("pm.train ",pm.train," pm.test ",pm.test," pm.val: ",pm.val)
"""