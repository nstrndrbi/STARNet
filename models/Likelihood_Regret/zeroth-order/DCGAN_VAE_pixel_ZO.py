import math
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

# Notes:  The Encoder class is a neural network module that represents an encoder. It takes an input image and transforms it into a latent representation.
# Here's a breakdown of its structure:
# The __init__ method initializes the Encoder class. It takes parameters such as isize (input size), 
# nz (size of the latent space), nc (number of input channels), 
# ngf (number of filters in the first layer), ngpu (number of GPUs to use), 
# and n_extra_layers (number of extra layers).
# The forward method takes an input tensor and performs the forward pass through the network. It uses convolutional layers, batch normalization, 
# and ReLU activation functions to transform the input image. It then computes the mean and logarithmic variance of the output using separate convolutional layers. 
# The reparametrize method is used to sample from the latent space using the mean and variance.
# The reparametrize method takes the mean and logarithmic variance as input and applies a reparameterization trick to sample from the latent space.



class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        n = math.log2(isize)
        assert n==round(n),'imageSize must be a power of 2'
        n=int(n)
        
        main = nn.Sequential()
        main.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        main.add_module('input-BN', nn.BatchNorm2d(ngf))
        main.add_module('input-relu',nn.ReLU(True))
        
        for i in range(n-3):
            # state size. (ngf) x 32 x 32
            main.add_module('pyramid:{0}-{1}:conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            main.add_module('pyramid:{0}:relu'.format(ngf * 2**(i+1)), nn.ReLU(True))
        # self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        # self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        ## for Point-Fusion
        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 3)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 3)

        self.main = main
    def reparametrize(self,mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        # print('This is output.shape:')    
        # print(output.shape)
        mu = self.conv1(output)
        logvar = self.conv2(output)
        z = self.reparametrize(mu,logvar)
        return [z,mu,logvar]
    


# Notes: The DCGAN_G class represents a generator in a deep convolutional generative adversarial network (DCGAN). It generates images from a given latent space. 
# Here's an overview of its structure:
# The __init__ method initializes the DCGAN_G class. It takes parameters such as isize (input size), 
# nz (size of the latent space), nc (number of output channels),
# ngf (number of filters in the first layer), and ngpu (number of GPUs to use).
# The forward method takes an input tensor and performs the forward pass through the network. 
# It uses transpose convolutional layers, batch normalization, and ReLU activation functions to generate an output image from the input tensor. 
# The output is reshaped and permuted to match the desired image dimensions.





################ Original One
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.isize = isize
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize= 4
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc*256, 4, 2, 1, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            #print("Shape after main:", output.shape)
            output = output.view(-1, self.nc, 256, self.isize, self.isize)
            #output = output.view(-1, self.nc*256, self.isize, self.isize)

            output = output.permute(0, 1, 3, 4, 2)
        else: 
            output = self.main(input)
            output = output.view(-1, self.nc, 256, self.isize, self.isize)
            #print("Shape after main:", output.shape)
            #output = output.view(-1, self.nc*256, self.isize, self.isize)

            output = output.permute(0, 1, 3, 4, 2)

        return output 







############### For features
# class DCGAN_G(nn.Module):
#     def __init__(self, isize, nz, nc, ngf, ngpu):
#         super(DCGAN_G, self).__init__()
#         self.ngpu = ngpu
#         self.nc = nc
#         self.isize = isize
#         assert isize % 16 == 0, "isize has to be a multiple of 16"

#         cngf, tisize = ngf//2, 4
#         while tisize != isize:
#             cngf = cngf * 2
#             tisize = tisize * 2

#         main = nn.Sequential()
#         # input is Z, going into a convolution
#         main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
#                         nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
#         main.add_module('initial:{0}:batchnorm'.format(cngf),
#                         nn.BatchNorm2d(cngf))
#         main.add_module('initial:{0}:relu'.format(cngf),
#                         nn.ReLU(True))

#         csize = 4
#         while csize < isize // 2:
#             main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf // 2),
#                             nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
#             main.add_module('pyramid:{0}:batchnorm'.format(cngf // 2),
#                             nn.BatchNorm2d(cngf // 2))
#             main.add_module('pyramid:{0}:relu'.format(cngf // 2),
#                             nn.ReLU(True))
#             cngf = cngf // 2
#             csize = csize * 2

#         # Adjusting the final layer to produce the desired output shape
#         main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
#                         nn.ConvTranspose2d(cngf, nc * 256, 3, 1, 1, bias=False))
#         self.main = main

#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#             output = output.view(-1, self.nc, 256, self.isize, self.isize)
#             output = output.permute(0, 1, 3, 4, 2)
#         else:
#             output = self.main(input)
#             output = output.view(-1, self.nc, 40, 40)
#             #output = output.permute(0, 1, 3, 4, 2)

#         return output

