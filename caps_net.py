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

    def forward(self,x):
        out = self.conv1(x)
        out = func.relu(out)

        out1= torch.reshape(out,[-1,self.chan2,8,20,20])
        out1 = self.prim_layer(out1)
        out1 = func.relu(out1)
        #[32,8,6,6]
        # out=func.relu(out)
        out1 =  out1.transpose(2,-1)
        out1= torch.reshape(out1,[-1,1152,1,8])
        # out1 = torch.matmul(self.dig_W,out1)

        dig_out = out1.matmul(self.dig_W).squeeze(2)
        dig_out = dig_out + self.dig_Wb
        # for i in range(1152):
        out = self.dynamic_routing(dig_out)
        #     dig_out[:,i,:]= torch.matmul(out1[:,i,..., 0], self.dig_W[..., i])
        # dig_out=func.leaky_relu(dig_out).transpose(1,-1)
        # dig_out = dig_out.matmul(self.dig_2)
        # dig_out =torch.sum(dig_out ,self.dig_Wb)

        # out = func.leaky_relu(dig_out).transpose(1,-1)


        # dig_out = self.digit_caps(out1)

        # out = dig_out.reshape([-1,10,16,1])
        # out= func.tanh(dig_out)
        out = self.output(out.unsqueeze(-1))
        out = out.reshape([-1,10])
        out = func.softmax(out)
        return out

    def dynamic_routing(self,u):
        batch_nr= u.shape[0]
        b=torch.tensor([[[0]*10]*1152]*batch_nr).float()
        s = torch.tensor([[[0]*10]*16]*batch_nr).float()
        for i in range(3):
            c = func.softmax(b,dim=1)
            for i in range(10):
                # for j in range(batch_nr):
                s[...,i]=torch.matmul(c[...,i].unsqueeze(1),u).squeeze(1)
            v = self.squash(s)
        return v


    def squash(self,s):
        #TODO
        s_l2 = torch.norm(s,dim=1).unsqueeze(-1)
        s_l1 = torch.sum(torch.abs(s),dim=1).unsqueeze(-1)
        v = s_l2/(1+s_l2)*(s.transpose(1,-1)/s_l1)
        return v