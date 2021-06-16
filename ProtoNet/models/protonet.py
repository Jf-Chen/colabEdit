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
        embeddings = self.encoder(images)  #torch.Size([80,640])
        #相当于reshape,5*(1+15)，640
        # view(dim,-1),-1的位置会自动计算应有的大小，
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1) 

        support_embeddings = embeddings[:self.args.N * self.args.K, :] # [5,640]
        query_embeddings = embeddings[self.args.N * self.args.K:, :] # [75,640]

        # squeeze(),移除所有大小为1的维度
        # unsqueeze(x),在第x维插入一个维度，比如现在相当于变成1*f_class
        mask_task = self.f_task(support_embeddings, level='task').unsqueeze(0) # torch.Size([1, 640])->torch.Size([1, 1, 640])
        mask_class = self.f_class(support_embeddings, level='class').unsqueeze(0) # torch.Size([1, 5, 640]

        alpha = self.h(support_embeddings, level='balance').squeeze(0) # torch.Size([2]
        [alpha_task, alpha_class] = alpha

        
        masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1)
        prototypes_unnorm = torch.mean(masked_support_embeddings.view(self.args.K, self.args.N, -1), dim=0) # torch.Size([5, 640])
        # 视余弦距离不同而加权,权重经过exp
        dis=torch.zeros(self.args.N,self.args.K,device=self.args.devices[0])
        for i in range(self.args.N): #0~5
            temp_p=prototypes_unnorm[i,:]
            for j in  range(self.args.K): #0~5
                temp_s=masked_support_embeddings[i,j,:]
                dis[i,j]=torch.exp(torch.nn.functional.cosine_similarity(temp_p,temp_s,dim=0))
        
        sum_dis=torch.sum(dis,1)
        prototypes_dis=torch.zeros(prototypes_unnorm.size(),device=self.args.devices[0],dtype=embeddings.dtype)
        for i in range(self.args.N): #0~5
            for j in  range(self.args.K): #0~5
                prototypes_dis[i,:]=prototypes_dis[i,:]+(dis[i,j]/sum_dis[i])*masked_support_embeddings[i,j,:]
        
        
        prototypes = F.normalize(prototypes_dis, dim=1, p=2) # torch.Size([5, 640])

        masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1)

        #logits ->torch.Size([5, 75, 5] torch.bmm:两矩阵相乘 tensor.t():矩阵转置
        logits = torch.bmm(masked_query_embeddings, prototypes.t().unsqueeze(0).expand(self.args.N, -1, -1)) / self.args.tau 
        x = torch.arange(self.args.N).long().cuda(self.args.devices[0]) # torch.Size([5])
        collapsed_logits = logits[x, :, x].t() # torch.Size([75, 5])

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