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

        mask_task = self.f_task(support_embeddings, level='task').unsqueeze(0) # ([1, 1, 640])
        mask_class = self.f_class(support_embeddings, level='class').unsqueeze(0) # [1, 5, 640]

        alpha = self.h(support_embeddings, level='balance').squeeze(0) # torch.Size([2]
        [alpha_task, alpha_class] = alpha

        masked_support_embeddings = support_embeddings.view(self.args.K, self.args.N, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class * alpha_class) # torch.Size([5, 5, 640])
        prototypes_unnorm = torch.mean(masked_support_embeddings.view(self.args.K, self.args.N, -1), dim=0) # torch.Size([5, 640]) 
        prototypes = F.normalize(prototypes_unnorm, dim=1, p=2) # torch.Size([5, 640])
        
        # query.unsqueeze().expand():[50,640]->[1,50,640]->[5,50,640]
        masked_query_embeddings = query_embeddings.unsqueeze(0).expand(self.args.N, -1, -1) * \
            (1 + mask_task * alpha_task) * (1 + mask_class.transpose(0, 1) * alpha_class) # torch.Size([5, 50, 640])
        
        
        #==================start 原本的相似度计算方式，输出是[args.N,args.Q*args.K]===============
        # # logits ->torch.Size([5, 50, 5]) torch.bmm:两矩阵相乘 tensor.t():矩阵转置
        # # torch.bmm([b,h,w],[b,w,h])=[b,h,h]
        # # .t().unsqueeze().expand()，size的变化为
        # # [5,640]->[640,5]->[1, 640, 5]->[5, 640, 5]
        # logits = torch.bmm(masked_query_embeddings, prototypes.t().unsqueeze(0).expand(self.args.N, -1, -1)) / self.args.tau # ([5, 50, 5])
        # x = torch.arange(self.args.N).long().cuda(self.args.devices[0]) # torch.Size([5])
        # collapsed_logits = logits[x, :, x].t() # torch.Size([50, 5])
        # # [x, :, x]相当于，当x=1时，提取logits[1, :, 1],(比如[1,1,1],...[1,50,1]),
        # # 得到50个数[1,50]，累计5次得到[5,50] 含义是各个样本相对于类别的相似度
        #===================================== end ===============================================
        
        #==================start 使用glofa_image2class DN4的相似度计算方式,输出是[args.N,args.Q*args.K]===============
        Similarity_list = []
        for i in range(self.args.Q * self.args.K):
            query_sam=masked_query_embeddings[0,i,:]
            query_sam_v=query_sam.contiguous().view(-1,1)
            
            if torch.cuda.is_available():
                inner_sim = torch.zeros(1, self.args.N).cuda()
            
            for j in range(self.args.N):
                support_set_sam=masked_support_embeddings[j,:,:]
                B,C=support_set_sam.size()
                descriptor=1
                support_set_sam_v = support_set_sam.contiguous().view(descriptor,-1)
                support_set_sam_v_norm = F.normalize(support_set_sam_v, dim=1, p=2)
                # 由于特征维度是1,不能进行归一化
                innerproduct_matrix = query_sam_v@support_set_sam_v_norm
                topk_value, topk_index = torch.topk(innerproduct_matrix, self.args.neighbor_k, 1)
                inner_sim[0, j] = torch.sum(topk_value)
                
            Similarity_list.append(inner_sim)
            
        Similarity_list = torch.cat(Similarity_list, 0)
        collapsed_logits=Similarity_list
        #===================================== end ===============================================

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