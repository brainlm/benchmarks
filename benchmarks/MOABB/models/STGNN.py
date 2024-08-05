import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, global_mean_pool
import speechbrain as sb


class PositionEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=torch.nn.ReLU()):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            activation,
            torch.nn.Linear(out_dim, out_dim)
        )

    def forward(self, pos):
        return self.mlp(pos)


# GNN model
class GnnModel(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU()):
        super().__init__()
        # Spatial Graph convolution
        self.spatial_gnn = torch_geometric.nn.Sequential('x, edge_index', [
            (GCNConv(in_features, out_features, node_dim=1), 'x, edge_index -> x'),
            (activation, 'x -> x'),
            (GCNConv(out_features, out_features, node_dim=1), 'x, edge_index -> x'),
            (activation, 'x -> x'),
        ])

    def forward(self, x, edge_index):
        # permute x to [T, batch_size * num_nodes, in_channels]
        x = x.permute(2, 0, 1)  # (N, D, T) --> # (T, N, D)
        x = self.spatial_gnn(x=x, edge_index=edge_index)  # (T, N, D')
        x = x.permute(1, 0, 2)  # (N, T, D')
        return x


class STGNN(torch.nn.Module):
    """STGNN.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    dense_n_neurons: int
        Output layer shape
    activation_type: str
        Activation function of the hidden layers.
    """

    def __init__(
        self,
        input_shape,  # (1, T, C, 1)
        cnn_kernels,
        cnn_kernelsize,
        cnn_pool,
        cnn_pool_type="avg",
        graph_pool_type="avg",
        dropout=0.5,
        embed_dim=768,
        dense_n_neurons=4,
        activation_type="relu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            self.activation = torch.nn.GELU()
        elif activation_type == "elu":
            self.activation = torch.nn.ELU()
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()

        self.T = input_shape[1]
        self.graph_pool_type = graph_pool_type

        # Temporal convolutional module
        self.temporal_frontend = nn.Sequential(
            nn.Conv2d(1, cnn_kernels[0], kernel_size=(cnn_kernelsize[0], 1), padding='same'),
            nn.BatchNorm2d(cnn_kernels[0], momentum=0.01, affine=True),
            self.activation,
        )
        # position encoder
        self.pos_encoder = PositionEncoder(in_dim=3, out_dim=embed_dim, activation=self.activation)

        # Spatial Graph convolution
        self.spatial_gnn = GnnModel(in_features=cnn_kernels[0], out_features=embed_dim, activation=self.activation)
        if self.graph_pool_type == 'avg':
            self.global_pool = global_mean_pool
        elif self.graph_pool_type == 'attention':
            self.global_pool = AttentionPool(position_dim=3, projection_dim=embed_dim, activation=self.activation)

        self.conv_module = nn.Sequential(
            nn.Conv2d(cnn_kernels[0], cnn_kernels[1], kernel_size=(cnn_kernelsize[1], 1)),
            nn.BatchNorm2d(cnn_kernels[1], momentum=0.01, affine=True),
            nn.AvgPool2d(kernel_size=(cnn_pool[0], 1), stride=(cnn_pool[0], 1)) if cnn_pool_type == "avg" else nn.MaxPool2d(kernel_size=(cnn_pool[0], 1), stride=(cnn_pool[0], 1)),
            self.activation,
            nn.Dropout(p=dropout),

            nn.Conv2d(cnn_kernels[1], cnn_kernels[2], kernel_size=(cnn_kernelsize[2], 1)),
            nn.BatchNorm2d(cnn_kernels[2], momentum=0.01, affine=True),
            nn.AvgPool2d(kernel_size=(cnn_pool[1], 1), stride=(cnn_pool[1], 1)) if cnn_pool_type == "avg" else nn.MaxPool2d(kernel_size=(cnn_pool[1], 1), stride=(cnn_pool[1], 1)),
            self.activation,
            nn.Dropout(p=dropout),
        )

        # Shape of intermediate feature maps
        dummy_input = torch.zeros(1, 1, self.T, 1)
        tfront = self.temporal_frontend(dummy_input)
        conv_out = self.conv_module(tfront)
        in_channels = conv_out.shape[1] * conv_out.shape[2]

        self.dense_module = torch.nn.Sequential(
            torch.nn.Linear(in_channels, dense_n_neurons),
            torch.nn.LogSoftmax(dim=1)
        )
        self.flatten = nn.Flatten()

    def forward(self, graph):
        """Returns the output of the model.

        Arguments
        ---------
        graph.x : torch.Tensor (N, T)
            Input to GNN.
        edge_index : torch.Tensor
            Edge indices for GNN.
        batch : torch.Tensor
            Batch vector for GNN.
        """
        assert graph.x.shape[1] == self.T, f"Inconsistency, unexpect size of T: {self.T}"
        assert len(graph.x.shape) == 3, f"feature dim should 3, got: {graph.x.shape}"
        x = graph.x.unsqueeze(1)

        # Apply temporal convolutions
        x = self.temporal_frontend(x).squeeze()  # (N, 1, T, 1) --> (N, D, T')
        
        # Add positional encoding
        pos_encoding = self.pos_encoder(graph.pos).unsqueeze(-1)  # (N, 3) ---> (N, D, 1)
        x = x + pos_encoding  # (N, D, T')

        # Apply GNN message passing on every T step
        x = self.spatial_gnn(x=x, edge_index=graph.edge_index)  # (N, T', D')
        
        # Graph level pooling
        x = self.graph_pooler(x, graph.batch, positions=graph.pos)  # (B, D', T', 1)
        
        # Apply spatio-temporal conv module
        x = self.conv_module(x)
        
        # flatten + dense
        x = self.flatten(x)  # (B, -1)
        x = self.dense_module(x)  # (B, classes)
        return x

    def graph_pooler(self, x, batch, positions=None):
        # Graph level pooling
        if self.graph_pool_type == 'attention':
            assert positions is not None, "To use attention global pooling, provide the positions."
            return self.global_pool(x, batch, positions)

        N, T, D = x.shape
        x_reshaped = x.reshape(N, -1)
        pooled = self.global_pool(x_reshaped, batch)
        x = pooled.reshape(-1, T, D)
        return x


class AttentionPool(nn.Module):
    def __init__(self, position_dim=3, projection_dim=64, tem=1.0, activation=nn.ELU()):
        super().__init__()
        self.position_dim = position_dim
        self.projection_dim = projection_dim
        self.tem = tem

        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, projection_dim),
            activation,
            nn.Linear(projection_dim, projection_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, batch, positions):
        # Compute attention weights
        weights = self.position_mlp(positions)  # (N, D)
        weights = self.softmax(weights / self.tem)  # (N, D)

        # Apply attention and pool
        x = x * weights.unsqueeze(1)  # (N, T, D) * (N, 1, D) = (N, T, D)
        x = scatter(x, batch, dim=0, reduce='sum')  # (B, T, D)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, D, T,  1)
        return x
