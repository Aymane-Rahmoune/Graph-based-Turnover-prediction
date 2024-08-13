import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_directed_multigraph(transactions, node_attributes):
    """
    Create a directed multigraph from transaction data and node attributes.

    Parameters:
    - transactions: DataFrame containing transaction data.
    - node_attributes: DataFrame containing node attributes.

    Returns:
    - A nx.MultiDiGraph object with nodes and edges added.
    """
    # Combine IDs to find unique entities
    entities = pd.concat([transactions['ID'], transactions['COUNTERPARTY']]).unique()
    graph = nx.MultiDiGraph()
    
    # Add nodes with attributes
    for entity_id in entities:
        attributes = node_attributes[node_attributes['ID'] == entity_id].iloc[0].to_dict()
        graph.add_node(entity_id, **attributes)
    
    # Add edges with transaction amounts as attributes
    for _, row in transactions.iterrows():
        graph.add_edge(row['ID'], row['COUNTERPARTY'], amount=row['TX_AMOUNT'])
    
    return graph
    
def filter_nodes_by_turnover(graph, turnovers, threshold=100000):
    """
    Remove nodes from the graph where the turnover exceeds a certain threshold.

    Parameters:
    - graph: The nx.MultiDiGraph object.
    - turnovers: A DataFrame with 'ID' and turnover values.
    - threshold: The turnover threshold for filtering nodes.

    Returns:
    - The filtered nx.MultiDiGraph object.
    """
    nodes_to_remove = [node_id for node_id, turnover in turnovers.items() if turnover > threshold]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def prepare_tensor_data(graph):
    """
    Prepare tensors for node features and edge attributes from the graph.

    Parameters:
    - graph: The nx.MultiDiGraph object.

    Returns:
    - Tuple of tensors: (node_features_tensor, edge_index_tensor, edge_attr_tensor).
    """
    node_features = [[attrs[attr] for attr in list(attrs.keys())[1:]] for _, attrs in graph.nodes(data=True)]
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(graph.nodes())}
    edges = [(node_id_to_index[src], node_id_to_index[dest], attrs) for src, dest, attrs in graph.edges(data=True)]
    edge_index_list = [[src, dest] for src, dest, _ in edges]
    edge_attr_list = [[attrs['amount']] for _, _, attrs in edges]
    
    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)
    
    return node_features_tensor, edge_index_tensor, edge_attr_tensor


def add_targets_to_data(data, graph, targets):
    """
    Add target values (turnovers) to the data object.

    Parameters:
    - data: The PyTorch Geometric Data object.
    - graph: The nx.MultiDiGraph object.
    - targets: A Series or dict mapping node IDs to target values.

    Returns:
    - The updated PyTorch Geometric Data object with target values added.
    """
    index_mapping = {node_id: idx for idx, node_id in enumerate(graph.nodes())}
    targets_tensor = torch.full((data.num_nodes,), float('nan'), dtype=torch.float)
    for node_id, turnover in targets.items():
        if node_id in index_mapping:
            targets_tensor[index_mapping[node_id]] = turnover
    data.y = targets_tensor
    return data

def add_targets_to_data(data, graph, targets):
    """
    Add target values (turnovers) to the data object.

    Parameters:
    - data: The PyTorch Geometric Data object.
    - graph: The nx.MultiDiGraph object.
    - targets: A Series or dict mapping node IDs to target values.

    Returns:
    - The updated PyTorch Geometric Data object with target values added.
    """
    index_mapping = {node_id: idx for idx, node_id in enumerate(graph.nodes())}
    targets_tensor = torch.full((data.num_nodes,), float('nan'), dtype=torch.float)
    for node_id, turnover in targets.items():
        if node_id in index_mapping:
            targets_tensor[index_mapping[node_id]] = turnover
    data.y = targets_tensor
    return data

def create_graph_data_pipeline(transactions_df, node_attributes_df, target_column, threshold=10**5):
    """
    Full pipeline for creating a PyTorch Geometric data object from transaction data.

    Parameters:
    - transactions_df: DataFrame containing transaction data.
    - node_attributes_df: DataFrame containing node attributes and targets.
    - target_column: The column name in node_attributes_df that contains target values.

    Returns:
    - A PyTorch Geometric Data object.
    """
    # Separate target values and attributes
    targets = node_attributes_df.set_index('ID')[target_column]
    node_attributes_df = node_attributes_df.drop(columns=target_column)
    
    #------ Create the graph ------#
    graph = create_directed_multigraph(transactions_df, node_attributes_df)
    print("Directed multigraph created.\n")

    # Initial checks for data integrity
    assert len(graph.edges) == transactions_df.shape[0], "The number of edges does not match the number of transactions."
    entities = pd.concat([transactions_df['ID'], transactions_df['COUNTERPARTY']]).unique()
    assert len(graph.nodes) == len(entities), "The number of nodes does not match the number of unique entities."
    
    print('Initial checks passed, number of nodes and edges are as expected.')
    print(f' -   Number of nodes: {len(graph.nodes)}')
    print(f' -   Number of edges: {len(graph.edges)}\n')
    
    #------ Remove nodes with high turnover ------#
    graph = filter_nodes_by_turnover(graph, targets, threshold=threshold)
    print(f"Nodes with turnover above {threshold} removed:")
    print(f' -   Number of nodes: {len(graph.nodes)}')
    print(f' -   Number of edges: {len(graph.edges)}\n')
    
    #------ Prepare tensors for node features and edge attributes ------#
    node_features_tensor, edge_index_tensor, edge_attr_tensor = prepare_tensor_data(graph)
    print('Tensor data prepared.\n')

    #------ Create PyG data object ------#
    data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    print('Pytorch Geometric data object created.\n')

    #------Add target values to the data object ------#
    data = add_targets_to_data(data, graph, targets)
    print('Added target values to data object.\n')

    assert not torch.isnan(data.y).any(), "The target contains NaN values.\n"
    
    print("Data pipeline completed successfully.")
    return data

class DataPreprocessor:
    def __init__(self, data: Data):
        self.data = data

    def scale_features(self):
        scaler_features = StandardScaler()
        self.data.x[self.data.train_mask] = torch.tensor(
            scaler_features.fit_transform(
                self.data.x[self.data.train_mask]
            ), 
            dtype=torch.float
        )
        self.data.x[self.data.val_mask] = torch.tensor(
            scaler_features.transform(
                self.data.x[self.data.val_mask]
            ), 
            dtype=torch.float
        )

    def scale_edge_attributes(self):
        scaler_edge = MinMaxScaler(feature_range=(0,1))
        scaled_edge_attributes = scaler_edge.fit_transform(
            self.data.edge_attr.numpy()
        )
        self.data.edge_attr = torch.tensor(scaled_edge_attributes, dtype=torch.float)