from os.path import expanduser, join
import time
import numpy as np
from mnist import MNIST
import torch.nn as nn
import torch.nn.functional as func
import torch.optim  as optimize
from torch import tensor, cat
import torch
import cv2



class CapsNet(nn.Module):
    chan1 = 256
    def __init__(self):
        super(CapsNet,self).__init__()

        self.conv1 = nn.Conv2d(1,self.chan1,9)
        self.chan2=int(self.chan1/8)

        self.prim_layer = nn.Conv3d(self.chan2,self.chan2,[1,9,9],stride =[1,2,2])
        self.digit_caps = nn.Conv3d(32,10,[8,3,3],stride =[8,1,1])
        self.dig_W = torch.rand([1152,8,16], dtype=torch.float32,requires_grad=True)
        self.dig_Wb = torch.zeros([1152,16], dtype=torch.float32,requires_grad=True)
        self.dig_2 = torch.rand([1152 , 10], dtype=torch.float32,requires_grad=True)
        torch.nn.init.xavier_uniform(self.dig_W )
        torch.nn.init.xavier_uniform(self.dig_2 )
        self.output =nn.Conv2d(10,10,[16,1])
        self.lin1 = nn.Linear(160,512)
        self.lin2 = nn.Linear(512, 1024)
        self.lin3 = nn.Linear(1024, 784)

    def forward(self,x, testing =True):
        out = self.conv1(x)
        out = func.relu(out)

        out1= torch.reshape(out,[-1,self.chan2,8,20,20])
        out1 = self.prim_layer(out1)
        out1 = func.relu(out1)
        #[32,8,6,6]
        # out=func.relu(out)
        out1 =  out1.transpose(2,-1)
        out1= torch.reshape(out1,[-1,1152,1,8])

        dig_out = out1.matmul(self.dig_W).squeeze(2)
        dig_out = dig_out + self.dig_Wb

        dig_out = self.dynamic_routing(dig_out)
        reconstructed = None
        if not testing:
            reconstructed = self.reconstruct(dig_out)
        out = self.output(dig_out.transpose(1,-1).unsqueeze(-1))
        out = out.reshape([-1,10])
        out = func.softmax(out)
        return out, reconstructed

    def reconstruct(self, dig_caps):
        inp = dig_caps.reshape([-1,160])
        out = self.lin1(inp)
        out = func.relu(out)
        out = self.lin2(out)
        out = func.relu(out)
        out = self.lin3(out)
        out = func.sigmoid(out)
        out = out.reshape([-1,1,28,28])
        return out

    def dynamic_routing(self,u):
        batch_nr= u.shape[0]
        b=torch.tensor([[[0]*1152]*10]*batch_nr).float()
        s = torch.tensor([[[0]*10]*16]*batch_nr).float()
        for i in range(3):
            c = func.softmax(b,dim=1)#.transpose(1,-1)
            # for i in range(10):
                # for j in range(batch_nr):
            s=torch.matmul(c,u)
            v = self.squash(s)
            b=b+torch.matmul(u,v).transpose(1,-1)
        return v

    def squash(self,s):
        #TODO
        s_l2 = torch.norm(s,dim=1).unsqueeze(-1)
        s_l1 = torch.sum(torch.abs(s),dim=1).unsqueeze(-1)
        v = s_l2/(1+s_l2)*(s.transpose(1,-1)/s_l1)
        return v
