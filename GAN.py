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
from ONN import ONNLayer, ONNDecoder, transform, Decoder

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = ONNLayer(1,16)
        self.conv2 = ONNLayer(16,32)
        self.conv3 = ONNLayer(32,64)
        self.Linear1 = nn.Linear(64, 32)
        self.Linear2 = nn.Linear(32, 1)

    def forward(self, inp):
        out= self.conv1(inp)
        out= self.conv2(out)
        out= self.conv3(out)
        out = out.reshape([-1,64])
        out= self.Linear1(out)
        out = func.tanh(out)
        out= self.Linear2(out)
        return func.sigmoid(out)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.onn1 = ONNLayer(1,64)
        # self.conv2 = ONNLayer(16,32)
        # self.conv3 = ONNLayer(32,64)
        self.dec1 = Decoder(1,32)

        self.conv1 = nn.Conv2d(32,64,3,padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.dec2 = Decoder(64,128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64,3,padding=1)
        self.dec3 = Decoder(64,32)
        self.bn3 = nn.BatchNorm2d(32)
        # self.upsample2 = nn.ConvTranspose2d(32, 32, 3, dilation=1)
        self.conv3 = nn.Conv2d(32, 16,5,padding=0)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=0)

        self.lin_model = nn.Sequential(
            nn.Linear(100,256),
            nn.Tanh(),
            nn.Linear(256,512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, inp):
        # out= self.onn1(inp)
        # out= self.conv2(out)
        # out= self.conv3(out)
        out = self.dec1(inp)
        out=self.conv1(out)
        out = func.leaky_relu(out,negative_slope=0.2)
        out = self.bn1(out)
        out = self.dec2(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = func.leaky_relu(out,negative_slope=0.2)
        out = self.dec3(out)
        out = self.bn3(out)
        # out = self.upsample2(out)
        out = self.conv3(out)
        out = func.leaky_relu(out, negative_slope=0.2)
        out = self.conv4(out)
        # out = func.relu(out)
        out = func.tanh(out)
        return out
    # def forward(self,inp):
    #     inp = inp.view(-1,100)
    #     out = self.lin_model(inp)
    #     out = out.reshape([-1,1,28,28])
    #     return out

class GAN:
    save_path_gen = 'generator3.ckpt'
    save_path_disc = 'disciminator3.ckpt'

    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.device = torch.device("cuda")

        self.generator.to(self.device )
        self.discriminator.to(self.device )
        pytorch_total_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print('total params', pytorch_total_params)
        self.disc_loss_func = nn.BCELoss()
        self.gen_loss_func = nn.BCELoss()
        self.gen_optim = optimize.Adam(params=self.generator.parameters(), lr=0.001)

        self.disc_optim = optimize.Adam(params=self.discriminator.parameters(),lr=0.0001)

        try:
            self.generator.load_state_dict(torch.load(self.save_path_gen))
            print('successfully reloaded models')
        except Exception as e:
            print('failed to reload model')
            print(e)
        try:
            self.discriminator.load_state_dict(torch.load(self.save_path_disc))

            print('successfully reloaded models')
        except Exception as e:
            print('failed to reload model')
            print(e)

    def get_gen_images(self):
        gen_in = 2 * np.random.random_sample([100, 1,7,7])
        gen_in = tensor(gen_in).float().to(self.device )
        gen_images = self.generator(gen_in)
        return gen_images

    def train_generator(self):
        self.discriminator.eval()
        self.generator.train()
        self.gen_optim.zero_grad()

        for param in self.generator.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = False

        gen_images = self.get_gen_images()

        gen_disc_out = self.discriminator(gen_images)
        gen_shape = gen_images.shape[0]
        # all_ones = tensor([0] * gen_shape).float().to(self.device)
        gen_loss = -torch.sum(torch.log(gen_disc_out))/gen_shape
        # gen_loss =self.gen_loss_func(gen_disc_out, all_ones.view(-1,1))

        print('gen_loss', gen_loss)
        gen_loss.backward()
        self.gen_optim.step()



    def train_discrimintor(self, real_ims):

        self.generator.eval()
        self.discriminator.train()
        self.disc_optim.zero_grad()

        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = True

        gen_images = self.get_gen_images()
        real_ims= tensor(real_ims).to(self.device)
        real_and_gen = cat([gen_images.detach(), real_ims], dim=0).float()
        gen_shape = gen_images.shape[0]
        labels = tensor([0] * gen_shape + [1] * real_ims.shape[0]).float()
        labels = labels.to(self.device )
        i_shuffle = torch.randperm(labels.shape[0])
        labels = labels[i_shuffle].view(-1, 1)
        real_and_gen = real_and_gen[i_shuffle]
        disc_out = self.discriminator(real_and_gen)
        disc_loss = self.disc_loss_func(disc_out, labels)
        print('disc loss', disc_loss)
        for l, i in zip(labels[:5],disc_out[:5]):
            print(l,i)

        disc_loss.backward()
        self.disc_optim.step()

    def save_model(self):
        if self.save_path_gen is not None:
            torch.save(self.generator.state_dict(), self.save_path_gen)
            torch.save(self.discriminator.state_dict(), self.save_path_disc)
            print("Model saved in file: %s" % self.save_path_gen)
        else:
            print('did not save', self.save_path_gen)


    def test(self):
        gen_in = 2*np.random.random_sample([1, 1, 7,7])
        gen_in = tensor(gen_in).float()
        gen_images = self.generator(gen_in.to(self.device ))
        show_im =  (gen_images[0][0].detach().cpu().numpy() *127.5 +127.5).astype(np.uint8)
        show_im = cv2.resize(show_im,(300,300))
        cv2.imshow('generated',show_im)
        key = cv2.waitKey(20)
        return key


if __name__ == "__main__":
    base_path = expanduser('~/Data')
    mnist_data = MNIST(base_path)
    batch_size = 52
    train_data = mnist_data.load_training_in_batches(batch_size)
    test_data = mnist_data.load_training_in_batches(batch_size)
    im_path = join(base_path)
    # train_data.load()
    test_net=GAN()
    counter = 0
    for train_batch, test_batch in zip(train_data,test_data):
        if len(train_batch[0]) < 10:
            train_data = mnist_data.load_training_in_batches(batch_size)
            test_data = mnist_data.load_training_in_batches(batch_size)
            continue
        t_start = time.time()
        im,label = transform(train_batch[0],train_batch[1])
        # if counter <100:
        # if counter % 2 ==0:
        # test_net.train_discrimintor(im)
        test_net.train_generator()
        # im, label = transform(test_batch[0],test_batch[1])
        # error_test = test_net.test(im, label)
        key = test_net.test()
        if key ==27:
            break
        # print(counter,'taken ',time.time() - t_start)
        counter+=1
        if counter % 2 ==0:
            test_net.save_model()
