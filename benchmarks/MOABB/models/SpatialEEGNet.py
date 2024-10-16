"""EEGNet adapted to handle variable electrode configurations.

Base EEGNet implementation borrowed from Davide Borra's implementation in Speechbrain/benchmarks:

    EEGNet from https://doi.org/10.1088/1741-2552/aace8c.
    Shallow and lightweight convolutional neural network proposed for a general decoding of single-trial EEG signals.
    It was proposed for P300, error-related negativity, motor execution, motor imagery decoding.

Authors
 * Drew Wagner, 2024
"""

import speechbrain as sb
import torch
import torch.nn as nn
import torch_geometric.utils
from torch_geometric.data import Data


class node_wise(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x: Data):
        new_x = x.clone()
        new_x.x = self.func(x.x.unsqueeze(-1).unsqueeze(-1)).squeeze(-2)

        return new_x


class SpatialFocus(nn.Module):
    def __init__(self, projection_dim, position_dim=3, tau=1.0):
        super().__init__()
        self.projection_dim = projection_dim
        self.position_dim = position_dim
        self.tau = tau

        self.similarity_module = nn.Sequential(
            nn.Linear(position_dim, projection_dim),
            nn.ELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, batch: Data):
        positions = batch.pos
        x = batch.x

        weights = self.similarity_module(positions).div_(self.tau)
        weights = torch_geometric.utils.softmax(
            weights, index=batch.batch
        ).view(weights.shape[0], 1, weights.shape[1], 1)
        x = x.unsqueeze(-2) * weights
        x = torch_geometric.utils.scatter(
            x,
            batch.batch,
            dim=0,
            dim_size=batch.batch_size,
            reduce="sum",
        )
        return x


class SpatialEEGNet(nn.Module):
    """EEGNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_depth_multiplier: int
        Depth multiplier of the 2d spatial depthwise convolution.
    cnn_spatial_max_norm: float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_septemporal_depth_multiplier: int
        Depth multiplier of the 2d temporal separable convolution.
    cnn_septemporal_kernelsize: tuple
        Kernel size of the 2d temporal separable convolution.
    cnn_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        spatial_focus,
        C,
        T,
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(33, 1),
        cnn_spatial_depth_multiplier=2,
        cnn_spatial_max_norm=1.0,
        cnn_spatial_pool=(4, 1),
        cnn_septemporal_depth_multiplier=1,
        cnn_septemporal_point_kernels=None,
        cnn_septemporal_kernelsize=(17, 1),
        cnn_septemporal_pool=(8, 1),
        cnn_pool_type="avg",
        dropout=0.5,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation_type="elu",
    ):
        super().__init__()
        if activation_type == "gelu":
            activation = nn.GELU()
        elif activation_type == "elu":
            activation = nn.ELU()
        elif activation_type == "relu":
            activation = nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128  # sampling rate of the original publication (Hz)

        # CONVOLUTIONAL MODULE
        self.temporal_frontend = nn.Sequential()
        # Temporal convolution
        self.temporal_frontend.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )
        self.temporal_frontend.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels,
                momentum=0.01,
                affine=True,
            ),
        )
        self.spatial_focus = spatial_focus
        self.conv_module = nn.Sequential()
        # Spatial depthwise convolution
        cnn_spatial_kernels = (
            cnn_spatial_depth_multiplier * cnn_temporal_kernels
        )
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=cnn_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels,
                momentum=0.01,
                affine=True,
            ),
        )
        self.conv_module.add_module("act_1", activation)
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_spatial_pool,
                stride=cnn_spatial_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_1", nn.Dropout(p=dropout))

        # Temporal separable convolution
        cnn_septemporal_kernels = (
            cnn_spatial_kernels * cnn_septemporal_depth_multiplier
        )
        self.conv_module.add_module(
            "conv_2",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_spatial_kernels,
                out_channels=cnn_septemporal_kernels,
                kernel_size=cnn_septemporal_kernelsize,
                groups=cnn_spatial_kernels,
                padding="same",
                padding_mode="constant",
                bias=False,
                swap=True,
            ),
        )

        if cnn_septemporal_point_kernels is None:
            cnn_septemporal_point_kernels = cnn_septemporal_kernels

        self.conv_module.add_module(
            "conv_3",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_septemporal_kernels,
                out_channels=cnn_septemporal_point_kernels,
                kernel_size=(1, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_septemporal_point_kernels,
                momentum=0.01,
                affine=True,
            ),
        )
        self.conv_module.add_module("act_3", activation)
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_septemporal_pool,
                stride=cnn_septemporal_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_3", nn.Dropout(p=dropout))

        # Shape of intermediate feature maps
        dense_input_size = self._num_flat_features(T, C)
        # DENSE MODULE
        self.dense_module = nn.Sequential()
        self.dense_module.add_module(
            "flatten",
            nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", nn.LogSoftmax(dim=1))

    def _num_flat_features(self, T, C):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """
        x = self.temporal_frontend(
            torch.ones(
                (
                    1,
                    T,
                    C,
                    1,
                )
            )
        )
        x = self.conv_module(x)
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch_geometric.Data (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        # x is initially (b*c, t)
        x = node_wise(self.temporal_frontend)(x)  # returns as Data (b*c, t', d)
        x = self.spatial_focus(x)  # returns as Tensor (b, t', c', d)
        x = self.conv_module(x)  # returns as Tensor (b, t'', 1, d')
        x = self.dense_module(x)

        return x
