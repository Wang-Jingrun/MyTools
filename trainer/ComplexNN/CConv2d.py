import torch
import torch.nn as nn


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)

        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        return output