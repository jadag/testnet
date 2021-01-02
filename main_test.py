from os.path import expanduser, join
import time
from mnist import MNIST
import torch.nn as nn
import torch.nn.functional as func
import torch.optim  as optimize
from torch import tensor, cat, load, save
from caps_net import CapsNet

import numpy as np
import cv2

class BaselineModel(nn.Module):

    def __init__(self):
        super(BaselineModel,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32, 64,3)
        # self.conv3 = nn.Conv2d(64, 128,3)
        # self.conv4 = nn.Conv2d(128, 64,3)

        self.Linear1 = nn.Linear(1600, 640)
        self.Linear2 = nn.Linear(640, 10)

    def forward(self, input):
        out = self.conv1(input)
        out = func.relu(out)
        out = func.max_pool2d(out,2)
        out = self.conv2(out)
        out = func.relu(out)
        out = func.max_pool2d(out,2)
        # out = self.conv3(out)
        # out = func.relu(out)
        # out = func.max_pool2d(out,2)
        # out = self.conv4(out)
        # out = func.relu(out)
        # out = func.max_pool2d(out,2)
        out = out.reshape([-1,1600])
        out= self.Linear1(out )
        out = func.tanh(out)
        out= self.Linear2(out)
        # out = func.tanh(out)
        return func.softmax(out)

class ONNLayer(nn.Module):
    def __init__(self, chan_in, chan_out,poly_n = 3):
        super(ONNLayer, self).__init__()
        self.poly_n = poly_n
        self.conv = nn.Conv3d(chan_in, chan_out, [poly_n, 3, 3], stride=[poly_n, 1, 1])

    def expand_inp(self,inp):
        new_tensor = []
        im_to_clone = inp.unsqueeze(2)
        for j in range(self.poly_n):
            new_tensor.append(im_to_clone.clone()**(1+j))
        concated = cat(new_tensor,dim=2)
        return concated

    def forward(self, input):
        inp = self.expand_inp(input)
        out = self.conv(inp)
        out = out.squeeze(2)
        out = func.tanh(out)
        out = func.max_pool2d(out, 2)
        return out

class ONNDecoder(ONNLayer):
    def __init__(self,chan_in, chan_out,poly_n = 3):
        super(ONNDecoder,self).__init__(chan_in, chan_out,poly_n)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        inp = self.expand_inp(input)
        out = self.conv(inp)
        out = out.squeeze(2)
        out = func.tanh(out)
        out = self.upsample(out)

        return out


class Decoder(nn.Module):
    def __init__(self,chan_in, chan_out,poly_n = 3):
        super(Decoder,self).__init__()
        self.upsample = nn.ConvTranspose2d(chan_out,chan_out,2,stride=2)
        self.conv = nn.Conv2d(chan_in,chan_out,3,padding=1)
        self.bn1 = nn.BatchNorm2d(chan_in)
        self.bn2 = nn.BatchNorm2d(chan_out)

    def forward(self, input):
        out = self.bn1(input)
        out = self.conv(out)
        out = out.squeeze(2)
        out = func.relu(out)
        out = self.bn2(out)
        out = self.upsample(out)

        return out


class SelfONN(nn.Module):
    def __init__(self):
        super(SelfONN, self).__init__()
        self.conv1 = ONNLayer(1,32)
        self.conv2 = ONNLayer(32,64)
        # self.conv3 = ONNLayer(32,64)
        self.Linear1 = nn.Linear(1600, 640)
        self.Linear2 = nn.Linear(640, 10)

    def forward(self, inp):
        out= self.conv1(inp)
        out= self.conv2(out)
        # out= self.conv3(out)
        out = out.reshape([-1,1600])
        out= self.Linear1(out)
        out = func.tanh(out)
        out= self.Linear2(out)
        return func.softmax(out)

class TestNet:
    save_path = 'model_save.ckpt'

    def __init__(self):
        self.model = CapsNet()

        for module, parameters in zip(self.model.modules(),self.model.parameters()):
            print(module,parameters.shape)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('total params', pytorch_total_params)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optim = optimize.Adam(params=self.model.parameters())
        self.load_model()

    def load_model(self):
        try:
            self.module.load_state_dict(load(self.save_path))
            print('successfully reloaded models')
        except Exception as e:
            print('failed to reload model from ', self.save_path)
            print(e)

    def train(self, input,labels):
        self.model.train()
        self.optim.zero_grad()
        input = tensor(input)
        labels = tensor(labels)
        predict = self.model(input)
        loss = self.loss_func(predict,labels)
        loss.backward()
        self.optim.step()
        print(loss)
        self.save_model()

    def test(self, input,labels):
        self.model.eval()

        input_tens = tensor(input)
        predict = self.model(input_tens)
        # print(predict)
        error =np.sum( np.argmax(labels,axis=1) != np.argmax(predict.detach().numpy(),axis=1))
        return error

    def save_model(self):
        if self.save_path is not None:
            save(self.model.state_dict(), self.save_path)

            print("Model saved in file: %s" % self.save_path)
        else:
            print('did not save', self.save_path)


class TestNetCaps(TestNet):
    def __init__(self):
        super(TestNetCaps,self).__init__()
        self.model = CapsNet()
        self.save_path = 'caps_net.ckpt'
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optim = optimize.Adam(params=self.model.parameters())
        self.reconstruct_loss = nn.BCEWithLogitsLoss()
        self.load_model()

    def train(self, input,labels):
        self.model.train()
        self.optim.zero_grad()
        input = tensor(input)
        labels = tensor(labels)
        predict, reconstruction = self.model(input, testing=False)
        loss = self.loss_func(predict,labels)
        loss_recn = self.reconstruct_loss(reconstruction,input )
        total_loass = loss+loss_recn
        total_loass.backward()
        self.optim.step()
        print(loss)
        self.save_model()


    def test(self, input,labels):
        self.model.eval()

        input_tens = tensor(input)
        predict, _ = self.model(input_tens)
        # print(predict)
        error =np.sum( np.argmax(labels,axis=1) != np.argmax(predict.detach().numpy(),axis=1))
        return error

def transform(im, labels):
    new_ims = []
    new_labels = []
    for i,l in zip(im,labels):
        new_im = np.array(i,dtype=np.float32).reshape([1,1,28,28])
        new_im = new_im/255
        new_ims.append(new_im)
        new_l = [[0]*10]
        new_l[0][l] =1
        new_labels.append(new_l)
    np_ims = np.concatenate(new_ims,axis=0)
    np_labels = np.concatenate(new_labels,axis=0).astype(np.float32)
    return np_ims, np_labels

if __name__ == "__main__":
    base_path = expanduser('~/Data')
    mnist_data = MNIST(base_path)
    batch_size = 100
    train_data = mnist_data.load_training_in_batches(batch_size)
    test_data = mnist_data.load_training_in_batches(batch_size)
    im_path = join(base_path)
    # train_data.load()
    test_net=TestNetCaps()
    counter = 0

    for train_batch, test_batch in zip(train_data,test_data):
        t_start = time.time()
        im,label = transform(train_batch[0],train_batch[1])
        test_net.train(im, label)
        im, label = transform(test_batch[0],test_batch[1])
        error_test = test_net.test(im, label)

        print(counter,'test error',error_test,'taken ',time.time() - t_start)
        counter+=1

