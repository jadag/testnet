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
from main_test import ONNLayer, ONNDecoder, transform, Decoder

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

class DCDiscrimintator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 5,stride=2,padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5,stride=2,padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5,stride=2,padding=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.Linear1 = nn.Linear(128, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 1)

    def forward(self, inp):
        out = self.conv1(inp)
        out = func.leaky_relu(out, negative_slope=0.2)
        out = self.bn1(out)
        out = self.conv2(out)
        out = func.leaky_relu(out, negative_slope=0.2)
        out = self.bn2(out)
        out = self.conv3(out)
        out = func.leaky_relu(out, negative_slope=0.2)
        out = self.bn3(out)
        out = self.conv4(out)
        out = func.leaky_relu(out, negative_slope=0.2)
        out = self.bn4(out)

        out = out.reshape([-1, 128])
        out = self.Linear1(out)
        out = func.tanh(out)
        out = self.Linear2(out)
        out = func.tanh(out)
        out = self.Linear3(out)
        out = func.tanh(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin = nn.Linear(128,8192,bias=False)
        self.dec1 = Decoder(256,128)
        self.dec2 = Decoder(128,64)
        self.dec3 = Decoder(64, 32)
        self.conv1 = nn.Conv2d(32,3,5,padding=2,bias=False)


    def forward(self, inp):
        out= self.lin(inp)
        out = func.tanh(out)
        out = out.reshape([-1,256,4,4])
        out= self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.conv1(out)
        out = func.tanh(out)
        return out


class GAN:

    save_path_gen = 'generator.ckpt'
    save_path_disc = 'disciminator.ckpt'

    def __init__(self):
        self.generator = Generator()
        self.discriminator = DCDiscrimintator()
        if torch.cuda.device_count():
            self.device = torch.device("cuda")

        self.generator.to(self.device )
        self.discriminator.to(self.device )
        pytorch_total_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print('total params', pytorch_total_params)
        self.disc_loss_func = nn.BCELoss()
        self.gen_loss_func = nn.BCELoss()

        self.gen_optim = optimize.Adam(params=self.generator.parameters(), lr=0.0001,weight_decay=0.001,betas=(0.45 ,0.55))
        self.disc_optim = optimize.Adam(params=self.discriminator.parameters(),lr=0.0003,weight_decay=0.0001)

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

    def sample_generator_input(self, nr_inputs = 1):
        gen_in = 2 * np.random.random_sample([nr_inputs, 128]) - 1
        gen_in = tensor(gen_in).float().to(self.device)
        return gen_in

    def get_gen_images(self,nr_inputs ):
        gen_in = self.sample_generator_input(nr_inputs)
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

        gen_images = self.get_gen_images(100)

        gen_disc_out = self.discriminator(gen_images)
        gen_shape = gen_images.shape[0]
        all_ones = tensor([1] * gen_shape).float().to(self.device)
        gen_loss = torch.mean(torch.log(all_ones - gen_disc_out))
        # gen_loss =self.gen_loss_func(gen_disc_out, all_ones.view(-1,1))
        id_as_fake = torch.where(gen_disc_out < -0.7)[0].shape[0]

        print('gen_loss', gen_loss,'id_as_fake',id_as_fake)
        gen_loss.backward()
        self.gen_optim.step()

        return id_as_fake


    def train_discrimintor(self, real_ims):

        self.generator.eval()
        self.discriminator.train()
        self.disc_optim.zero_grad()

        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.discriminator.parameters():
            param.requires_grad = True

        gen_images = self.get_gen_images(100)
        real_ims= tensor(real_ims).to(self.device)
        real_and_gen = cat([gen_images.detach(), real_ims], dim=0).float()
        gen_shape = gen_images.shape[0]
        labels = tensor([-1] * gen_shape + [1] * real_ims.shape[0]).float()
        labels = labels.to(self.device )
        i_shuffle = torch.randperm(labels.shape[0])
        labels = labels[i_shuffle].view(-1, 1)
        real_and_gen = real_and_gen[i_shuffle]
        disc_out = self.discriminator(real_and_gen)
        disc_loss = torch.mean((disc_out - labels)**2)
        print('disc loss', disc_loss)
        id_as_fake = torch.where(disc_out<-0.7)[0].detach().cpu().numpy()
        are_fake = torch.where(labels<-0.7)[0].detach().cpu().numpy()
        id_as_real = torch.where(disc_out > 0.7)[0].detach().cpu().numpy()
        are_real = torch.where(labels > 0.7)[0].detach().cpu().numpy()
        false_neg = [x for x in id_as_fake if x not in are_real]
        true_neg = [x for x in id_as_fake if x in are_fake]
        false_pos = [x for x in id_as_real if x not in are_real]
        true_pos = [x for x in id_as_real if x in are_real]

        accuracy = len(true_neg+true_pos)/200
        for l, i in zip(labels[:5],disc_out[:5]):
            print(l,i)

        disc_loss.backward()
        if accuracy < 0.94:
            self.disc_optim.step()
            print("updated weights")
        print('accuracy',accuracy)
        return accuracy

    def save_model(self):
        if self.save_path_gen is not None:
            torch.save(self.generator.state_dict(), self.save_path_gen)
            torch.save(self.discriminator.state_dict(), self.save_path_disc)
            print("Model saved in file: %s" % self.save_path_gen)
        else:
            print('did not save', self.save_path_gen)


    def test(self):
        gen_images = self.get_gen_images(nr_inputs=3)
        show_im =  ((gen_images.cpu().detach().numpy()+1)*127.5).astype(np.uint8)
        for i, im in enumerate(show_im):
            show_im = cv2.resize(im.swapaxes(0,-1),(300,300))
            cv2.imshow('generated'+str(i),show_im)
        key = cv2.waitKey(20)
        return key


if __name__ == "__main__":
    base_path = expanduser('~/Data')
    mnist_data = MNIST(base_path)
    batch_size = 100
    train_data = mnist_data.load_training_in_batches(batch_size)
    test_data = mnist_data.load_training_in_batches(batch_size)
    im_path = join(base_path)
    test_net=GAN()
    counter = 0
    accuracy = 0
    id_as_fake = 0
    for train_batch, test_batch in zip(train_data,test_data):
        if len(train_batch[0]) < 10:
            train_data = mnist_data.load_training_in_batches(batch_size)
            test_data = mnist_data.load_training_in_batches(batch_size)
            continue
        t_start = time.time()
        im,label = transform(train_batch[0],train_batch[1])


        accuracy = test_net.train_discrimintor(im)
        while accuracy < 0.75:
            accuracy = test_net.train_discrimintor(im)
        id_as_fake = test_net.train_generator()
        key = test_net.test()
        if key ==27:
            break

        counter+=1
        if counter % 2 ==0:
            test_net.save_model()
