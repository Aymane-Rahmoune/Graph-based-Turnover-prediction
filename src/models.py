from typing import Optional, Dict
import torch_geometric.data as Data

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, global_add_pool, GatedGraphConv, GCNConv
from torch_geometric.nn import BatchNorm


class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features: int, h_dim: int = 10, dropout: float = 0.1, output_dim: int = 1):
        super(SimpleGCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, h_dim)  # First GCN layer
        self.bn1 = torch.nn.BatchNorm1d(h_dim)  # Batch Normalization for first layer
        
        self.conv2 = GCNConv(h_dim, h_dim)  # Output layer
        self.bn2 = torch.nn.BatchNorm1d(h_dim)  # Batch Normalization for output layer
        self.dropout = torch.nn.Dropout(dropout)  # Dropout layer
        
        self.linear1 = torch.nn.Linear(h_dim, h_dim)  # Adding a linear layer that maps from output_dim to output_dim
        self.bn3 = torch.nn.BatchNorm1d(h_dim)
        self.linear2 = torch.nn.Linear(h_dim, output_dim)

        self.name = f'GCN--h_dim_{h_dim}--dropout_{dropout}'


    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.linear2(x)

        return x.squeeze()


class GATModel(torch.nn.Module):
    def __init__(self, num_node_features: int, h_dim: int = 4, heads: int = 4, dropout: float = 0.2, output_dim: int = 1):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(num_node_features, h_dim, heads=heads, concat=False, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(h_dim)
        
        self.gat2 = GATConv(h_dim, h_dim, heads=heads, concat=False)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)
        
        self.linear1 = torch.nn.Linear(h_dim, h_dim)
        self.bn3 = torch.nn.BatchNorm1d(h_dim)
        self.linear2 = torch.nn.Linear(h_dim, output_dim)
        
        self.name = f'GAT--h_dim_{h_dim}--heads_{heads}--dropout_{dropout}'

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.gat1(x, edge_index, edge_weight))
        x = self.bn1(x)

        x = self.gat2(x, edge_index, edge_weight)
        x = self.bn2(x)
        
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.linear2(x)
        
        return x.squeeze()


class GIN(torch.nn.Module):
    def __init__(self, num_features: int, dim: int, output_dim: int, dropout_rate: float = 0.1):
        super(GIN, self).__init__()
        nn1 = Sequential(
            Linear(num_features, dim), 
            ReLU(), Linear(dim, dim), 
            Dropout(dropout_rate)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(dim)

        nn2 = Sequential(
            Linear(dim, dim), 
            ReLU(), 
            Linear(dim, dim), 
            Dropout(dropout_rate)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.dropout = Dropout(dropout_rate)  
        self.fc2 = Linear(dim, output_dim)

        self.name = f'GIN--h_dim_{dim}--dropout_{dropout_rate}'

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatedGCNModel(torch.nn.Module):
    def __init__(self, h_dim: int = 4, dropout: float = 0.2, output_dim: int = 1):
        super(GatedGCNModel, self).__init__()
        self.ggc1 = GatedGraphConv(out_channels=h_dim, num_layers=1)
        self.ggc2 = GatedGraphConv(out_channels=h_dim, num_layers=1)

        # Linear layers
        self.linear1 = torch.nn.Linear(h_dim, h_dim)
        self.linear2 = torch.nn.Linear(h_dim, output_dim)

        # Batch normalization layers
        self.bn1 = BatchNorm(h_dim)
        self.bn2 = BatchNorm(h_dim)
        self.bn3 = BatchNorm(h_dim)

        self.name = f'GatedGCN--h_dim_{h_dim}--dropout_{dropout}'

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.ggc1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.relu(self.ggc2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.relu(self.linear1(x))
        x = self.bn3(x)
        x = self.linear2(x)
        return x.squeeze()


def initialize_model(model_name: str, model_config: Dict, num_node_features: int, num_edge_features: int) -> torch.nn.Module:
    """
    Initialize a model based on its name and configuration.

    Parameters:
    - model_name (str): The name of the model to initialize.
    - model_config (dict): Configuration parameters for the model.
    - num_node_features (int): The number of features for each node in the graph.
    - num_edge_features (int): The number of features for each edge in the graph.

    Returns:
    - model (torch.nn.Module): The initialized model.
    """
    if model_name == "SimpleGCN":
        model = SimpleGCN(num_node_features=num_node_features,
                          h_dim=model_config.get('h_dim', 10),
                          dropout=model_config.get('dropout', 0.1),
                          output_dim=model_config.get('output_dim', 1))
    
    elif model_name == "GATModel":
        model = GATModel(num_node_features=num_node_features,
                         h_dim=model_config.get('h_dim', 4),
                         heads=model_config.get('heads', 4),
                         concat=model_config.get('concat', False),
                         dropout=model_config.get('dropout', 0.2),
                         output_dim=model_config.get('output_dim', 1))
    
    elif model_name == "GIN":
        model = GIN(num_features=num_node_features,
                    dim=model_config.get('dim', 32),
                    output_dim=model_config.get('output_dim', 1),
                    dropout_rate=model_config.get('dropout_rate', 0.1))
    
    elif model_name == "GatedGCNModel":
        model = GatedGCNModel(h_dim=model_config.get('h_dim', 4),
                              dropout=model_config.get('dropout', 0.2),
                              output_dim=model_config.get('output_dim', 1))
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model
