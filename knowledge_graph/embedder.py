import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.utils import from_networkx

class GraphEmbedder(nn.Module):
    """
    Learns node embeddings from a graph snapshot using Graph Convolutional Networks (GCN).
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # 2 layers of GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix (num_nodes, in_channels)
        edge_index: Graph connectivity (2, num_edges)
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

    @staticmethod
    def nx_to_pyg(graph: nx.MultiDiGraph, feature_dim: int) -> Data:
        """
        Helper utility to convert a networkx snapshot into a torch_geometric Data object.
        Initializes dummy features (e.g., ones) of shape (num_nodes, feature_dim) if real features aren't present.
        """
        # Convert to simple directed graph as pyg may have issues with MultiDiGraph from_networkx
        di_graph = nx.DiGraph(graph)
        data = from_networkx(di_graph)
        
        # Initialize dummy node features if missing
        if not hasattr(data, 'x') or data.x is None:
            num_nodes = di_graph.number_of_nodes()
            data.x = torch.ones((num_nodes, feature_dim), dtype=torch.float)
            
        return data
