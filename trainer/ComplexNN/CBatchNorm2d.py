import torch
import torch.nn as nn


class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                     affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                   affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)

        output = torch.stack([n_real, n_im], dim=-1)
        return output