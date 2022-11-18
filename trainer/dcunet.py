from .ComplexNN import *


class Encoder(nn.Module):
    """
    Class of upsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted


class Decoder(nn.Module):
    """
    Class of downsample block
    """

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.filter_size, stride=self.stride_size,
                                       output_padding=self.output_padding, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            mag = torch.abs(conved)
            m_phase = conved / (mag + 1e-8)
            m_mag = torch.tanh(mag)
            output = m_phase * m_mag

        return output


class DCUnet10(nn.Module):
    """
    Deep Complex U-Net class of the model.
    """

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90)

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=90, output_padding=(0, 1),
                                 out_channels=1, last_layer=True)

    def forward(self, x, is_istft=True):
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # upsampling/decoding
        u0 = self.upsample0(d4)
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # u4 - the mask
        output = u4 * x
        if is_istft:
            output = torch.squeeze(output, 1)
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        return output