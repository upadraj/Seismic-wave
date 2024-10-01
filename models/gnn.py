import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Define the GNN model for seismic wave classification
class SeismicGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SeismicGNN, self).__init__()
        # Define two GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # node features and edge index

        # First Graph Convolutional Layer + Activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply ReLU non-linearity

        # Second Graph Convolutional Layer (output layer)
        x = self.conv2(x, edge_index)

        # Apply softmax activation to get class probabilities
        return F.log_softmax(x, dim=1)
