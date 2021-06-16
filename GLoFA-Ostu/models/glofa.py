# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2021-04-06 19:31:25
"""

import torch
from torch import nn
from torch.nn import functional as F

from networks.set_function import SetFunction

class MyModel(nn.Module):
    def __init__(self, args, network):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
        if args.network_name == 'resnet':
            dimension = 640
        self.f_task = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.f_class = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
        self.h = SetFunction(args, input_dimension=dimension, output_dimension=2)
        
        # f_task :  cuda:0 SetFunction(
          # (psi): Sequential(
            # (0): Linear(in_features=640, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=1280, bias=True)
            # (3): ReLU()
          # )
          # (rho): Sequential(
            # (0): Linear(in_features=1920, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=640, bias=True)
          # )
        # )
        # f_class :  cuda:0 SetFunction(
          # (psi): Sequential(
            # (0): Linear(in_features=640, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=1280, bias=True)
            # (3): ReLU()
          # )
          # (rho): Sequential(
            # (0): Linear(in_features=1920, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=640, bias=True)
          # )
        # )
        # h :  cuda:0 SetFunction(
          # (psi): Sequential(
            # (0): Linear(in_features=640, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=1280, bias=True)
            # (3): ReLU()
          # )
          # (rho): Sequential(
            # (0): Linear(in_features=1920, out_features=1280, bias=True)
            # (1): ReLU()
            # (2): Linear(in_features=1280, out_features=2, bias=True)
          # )
        # )
        
    
    def forward(self, images):
        # 假设是5-way 5-shot,N=5,K=5,Q=10
        # images->torch.Size([75, 3, 84, 84])
        embeddings = self.encoder(images)  #torch.Size([75,640])
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)  # [75,640]

        support_embeddings = embeddings[:self.args.N * self.args.K, :] # [25,640]
        query_embeddings = embeddings[self.args.N * self.args.K:, :] # [50,640]
        
        
        # 对support和query相同处理，主要修改求mean的方式
            
            
        #—————————————————————— start —————————————————————————————#
        mask_task = self.f_task(support_embeddings, level='task').unsqueeze(0) # ([1, 1, 640])
        mask_class = self.f_class(support_embeddings, level='class').unsqueeze(0) # [1, 5, 640]

        alpha = self.h(support_embeddings, level='balance').squeeze(0) # torch.Size([2]
        [alpha_task, alpha_class] = alpha

        # masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1) * \
            #(1 + mask_task * alpha_task) * (1 + mask_class * alpha_class) # torch.Size([5, 5, 640])
        masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1)
        prototypes_unnorm = torch.mean(masked_support_embeddings.view(self.args.K, self.args.N, -1), dim=0) # torch.Size([5, 640])
        epsilon=1;
        # 视余弦距离不同而加权
        dis=torch.zeros(self.args.N,self.args.K,device=self.args.devices[0])
        for i in range(masked_support_embeddings.size()[0]): #0~5
            temp_p=prototypes_unnorm[i,:]
            for j in  range(masked_support_embeddings.size()[1]): #0~5
                temp_s=masked_support_embeddings[i,j,:]
                dis[i,j]=torch.exp(torch.nn.functional.cosine_similarity(temp_p,temp_s,dim=0))
                
        sum_dis=torch.sum(dis,1)
        prototypes_dis=torch.zeros(prototypes_unnorm.size(),device=self.args.devices[0])
        for i in range(masked_support_embeddings.size()[0]): #0~5
            for j in  range(masked_support_embeddings.size()[1]): #0~5
                prototypes_dis[i,:]=prototypes_dis[i,:]+(dis[i,j]/sum_dis[i])*masked_support_embeddings[i,j,:]
        
        
        
        prototypes = F.normalize(prototypes_dis, dim=1, p=2) # torch.Size([5, 640])
        
        # query.unsqueeze().expand():[50,640]->[1,50,640]->[5,50,640]
        # masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1) * \
            # (1 + mask_task * alpha_task) * (1 + mask_class.transpose(0, 1) * alpha_class) # torch.Size([5, 50, 640])
        masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1)
        #------------------------- end ---------------------------------#
        
        
        # logits ->torch.Size([5, 50, 5]) torch.bmm:两矩阵相乘 tensor.t():矩阵转置
        # torch.bmm([b,h,w],[b,w,h])=[b,h,h]
        # .t().unsqueeze().expand()，size的变化为
        # [5,640]->[640,5]->[1, 640, 5]->[5, 640, 5]

        #logits ->torch.Size([5, 75, 5] torch.bmm:两矩阵相乘 tensor.t():矩阵转置
        logits = torch.bmm(masked_query_embeddings, prototypes.t().unsqueeze(0).expand(self.args.N, -1, -1)) / self.args.tau  # ([5, 50, 5])
        x = torch.arange(self.args.N).long().cuda(self.args.devices[0]) # torch.Size([5])
        collapsed_logits = logits[x, :, x].t() # torch.Size([50, 5])
        # [x, :, x]相当于，当x=1时，提取logits[1, :, 1],(比如[1,1,1],...[1,50,1]),
        # 得到50个数[1,50]，累计5次得到[5,50] 含义是各个样本相对于类别的相似度

        return collapsed_logits
    
    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
    
    def get_other_params(self):
        modules = [self.f_task, self.f_class, self.h]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j
