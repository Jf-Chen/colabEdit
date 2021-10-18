### my_finetune注释



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

args.train_way=5 # 原本是20

args.test_shot=1,5

args.pre=True

args.gpu=0

**其中  decay_epoch，val_epoch，weight_decay，nesterov意义不明，**

**--train_query_shot=15，--test_query_shot不知道作用**

