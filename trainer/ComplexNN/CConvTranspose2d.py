import torch
import torch.nn as nn


class CConvTranspose2d(nn.Module):
    """
      Class of complex valued deconvolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=(0, 0), padding=(0, 0)):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output