# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-08-07 14:07:17
"""

import torch
from torch import nn
from torch.nn import functional as F

# 看作是MLP
class SetFunction(nn.Module):
    def __init__(self, args, input_dimension, output_dimension):
        super(SetFunction, self).__init__()# nn.Module.__init()__
        self.args = args
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # dimension=640
        # SetFunction(
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
        self.psi = nn.Sequential(
            # nn.Linear() y=x*A'+b，A是weight，
            nn.Linear(input_dimension, input_dimension  * 2), 
            nn.ReLU(),
            nn.Linear(input_dimension * 2, input_dimension * 2),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(input_dimension * 3, input_dimension * 2),
            nn.ReLU(),
            nn.Linear(input_dimension * 2, output_dimension),
        )

    def forward(self, support_embeddings, level):
        if level == 'task':
            # support_embeddings [5,640]
            psi_output = self.psi(support_embeddings) #[5,1280]
            # torch.cat()两个张量连接在一起
            rho_input = torch.cat([psi_output, support_embeddings], dim=1) # [5,1920]
            rho_input = torch.sum(rho_input, dim=0, keepdim=True) # [1,1920]
            # relu6,就是普通的relu,但限制最大输出为6
            # rho_output = F.relu6(self.rho(rho_input)) / 6 * self.args.delta
            rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 * self.args.delta # [1,640]
            
            return rho_output
        elif level == 'class':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim=1)
            rho_input = torch.sum(rho_input.view(self.args.K, self.args.N, -1), dim=0)
            rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 * self.args.delta
            return rho_output
        elif level == 'balance':
            psi_output = self.psi(support_embeddings)
            rho_input = torch.cat([psi_output, support_embeddings], dim=1)
            rho_input = torch.sum(rho_input, dim=0, keepdim=True)
            rho_output = torch.nn.functional.relu(self.rho(rho_input))
            return rho_output