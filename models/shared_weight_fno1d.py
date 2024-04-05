import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.distributed.rpc import RRef

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        # out_ft = self.einsum.forward(x_ft[:, :, :self.modes1])
        out_ft = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.gelu = nn.GELU()

        self.mlp_block = nn.Sequential(
            self.mlp1,
            self.gelu,
            self.mlp2,
        )

    def forward(self, x):
        x = self.mlp_block(x)
        return x


class integral_kernel_block(nn.Module):

    def __init__(self, width, is_gelu=True):
        super(integral_kernel_block, self).__init__()
        self.is_gelu = is_gelu
        
        self.gelu = nn.GELU()
        self.mlp = MLP(width, width, width)
        self.w = nn.Conv1d(width, width, 1)

    def change_weights(self, num):
        nn.init.constant_(self.remote_conv.weights1, num)

    def forward(self, x_conv, x):
        x1 = self.mlp(x_conv)
        x2 = self.w(x)
        x = x1 + x2
        if self.is_gelu:
            x = self.gelu(x)
        
        return x

class preprocess_block(nn.Module):

    def __init__(self, width):
        super(preprocess_block, self).__init__()
        self.width = width
        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.p(x)
        x = x.permute(0, 2, 1)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class projection_block(nn.Module):

    def __init__(self, width):
        super(projection_block, self).__init__()
        self.width = width
        self.q = MLP(self.width, 1, self.width * 2)  # output channel_dim is 1: u1(x)
    
    def forward(self, x):
        
        x = self.q(x)
        x = x.permute(0, 2, 1)
        
        return x

class FNO1d(nn.Sequential):
    def __init__(self, modes, width, block_number=4):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        #self.padding = 8  # pad the domain if input is non-periodic
        self.remote_conv = SpectralConv1d(width, width, modes)

        self.add_module('linear', preprocess_block(self.width))
        
        for i in range(0,block_number-1):
          self.add_module('kernel_layer_'+str(i), integral_kernel_block(self.width, self.modes1, self.remote_conv, is_gelu=True))
        self.add_module('kernel_layer_'+str(block_number-1),integral_kernel_block(self.width, self.modes1, self.remote_conv, is_gelu=False))

        self.add_module('projection', projection_block(self.width)) 
    
    def is_spectral_weights_equal(self):
        print("compare data ptr ", self.kernel_layer_0.conv.einsum.weights1.data_ptr() == self.kernel_layer_1.conv.einsum.weights1.data_ptr())
        return torch.all((self.kernel_layer_0.remote_conv.weights1-self.kernel_layer_1.remote_conv.weights1) == 0)
    
    def print_spectral_weights(self):
        print(self.kernel_layer_0.remote_conv.weights1)
        print(self.kernel_layer_1.remote_conv.weights1)
        print(self.kernel_layer_2.remote_conv.weights1)


class ModelParallelFNO1d(nn.Module):
    def __init__(self, modes, width, devices):
        super(ModelParallelFNO1d, self).__init__()

        self.devices = devices
        self.modes1 = modes
        self.width = width

        self.remote_conv = SpectralConv1d(width, width, modes)
        self.linear = preprocess_block(self.width)

        self.kernel_layer_0 = integral_kernel_block(self.width, self.remote_conv)
        self.kernel_layer_1 = integral_kernel_block(self.width, self.remote_conv)
        self.kernel_layer_2 = integral_kernel_block(self.width, self.remote_conv)
        self.kernel_layer_3 = integral_kernel_block(self.width, self.remote_conv, is_gelu=False)

        self.projection = projection_block(self.width)

        self.seq1 = nn.Sequential(
            self.linear,
            self.kernel_layer_0,
            self.kernel_layer_1,
            self.kernel_layer_2,
            self.kernel_layer_3,
        ).cuda(self.devices[0])

        self.seq2 = nn.Sequential(
            
            self.projection,
        ).cuda(self.devices[1])

    def forward(self, x):
        x = self.seq2(self.seq1(x).to(self.devices[1]))
        return x

class ModelParallelFNO1d_rpc(nn.Module):
    def __init__(self, modes, width, devices, remote_conv):
        super(ModelParallelFNO1d_rpc, self).__init__()

        self.devices = devices
        self.modes1 = modes
        self.width = width

        self.linear = preprocess_block(self.width).cuda(self.devices[0])

        self.conv0 = remote_conv
        self.kernel_layer_0 = integral_kernel_block(self.width).cuda(self.devices[0])
        self.kernel_layer_1 = integral_kernel_block(self.width).cuda(self.devices[0])
        self.kernel_layer_2 = integral_kernel_block(self.width).cuda(self.devices[1])
        self.kernel_layer_3 = integral_kernel_block(self.width, is_gelu=False).cuda(self.devices[1])

        self.projection = projection_block(self.width).cuda(self.devices[1])


    def forward(self, x):
        x = x.cuda(self.devices[0])
        x = self.linear(x) 
        # x_rref = RRef(x)
        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_0(y.cuda(x.get_device()), x)
        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_1(y.cuda(x.get_device()), x)
        y = self.conv0.forward(x.cpu())

        x = x.cuda(self.devices[1])
        x = self.kernel_layer_2(y.cuda(self.devices[1]), x)
        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_3(y.cuda(x.get_device()), x)
        x = self.projection(x)

        return x
    
class FNO1d_rpc(nn.Module):
    def __init__(self, modes, width, rank, remote_conv):
        super(FNO1d_rpc, self).__init__()

        self.rank = rank
        self.modes1 = modes
        self.width = width

        self.linear = preprocess_block(self.width).cuda(rank)

        self.conv0 = remote_conv
        self.kernel_layer_0 = integral_kernel_block(self.width).cuda(rank)
        self.kernel_layer_1 = integral_kernel_block(self.width).cuda(rank)
        self.kernel_layer_2 = integral_kernel_block(self.width).cuda(rank)
        self.kernel_layer_3 = integral_kernel_block(self.width, is_gelu=False).cuda(rank)

        self.projection = projection_block(self.width).cuda(rank)

    
    def forward(self, x):
        x = self.linear(x) 
        # x_rref = RRef(x)
        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_0(y.cuda(x.get_device()), x)

        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_1(y.cuda(x.get_device()), x)

        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_2(y.cuda(x.get_device()), x)

        y = self.conv0.forward(x.cpu())
        x = self.kernel_layer_3(y.cuda(x.get_device()), x)
        
        x = self.projection(x)
        return x

    

# m = FNO1d(16, 64)
# # print('named params')
# # for n, p in m.named_parameters():
# #     print(n, p.shape, p.data_ptr)

# # print('params')
# # for p in m.parameters():
# #     print(p.shape, p.data_ptr)

# # print('state dict')
# # for n, p in m.state_dict().items():
# #     print(n, p.shape, p.data_ptr)

# # TO CHANGE WEIGHTS AND CHECK EQUALITY
# # before saving
# m.kernel_layer_0.change_weights(1)
# # m.print_spectral_weights()
# print(m.is_spectral_weights_equal())

# # save the model
# torch.save(m.state_dict(), 'sample.pth')

# new_m = FNO1d(16, 64)
# new_m.load_state_dict(torch.load('sample.pth'))

# # before saving
# new_m.kernel_layer_1.change_weights(2)
# # new_m.print_spectral_weights()
# print(new_m.is_spectral_weights_equal())