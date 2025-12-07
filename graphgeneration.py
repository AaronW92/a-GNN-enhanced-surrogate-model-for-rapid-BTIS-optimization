import pandas as pd
import uuid

# Read input Excel files
training_data = pd.read_excel("training_data.xlsx")
nodes_data = pd.read_excel("nodes.xlsx")
edges_data = pd.read_excel("edges.xlsx")

# Define mapping from node attributes to training data columns
attribute_mapping = {
    'A': 'PUR',
    'B': 'XPS_roof',
    'C': 'XPS_floor',
    'D': 'Mortar',
    'E': 'Silicene_wall',
    'F': 'Silicene_floor',
    'G': 'BW_Low-e',
    'H': 'EW_Low-e',
    'I': 'CW_Low-e',
    'none': 0.0
}

# Map '0' in nodes_data to 'none' for compatibility
nodes_data['wall'] = nodes_data['wall'].replace('0', 'none')
nodes_data['glass'] = nodes_data['glass'].replace('0', 'none')
nodes_data['ceiling'] = nodes_data['ceiling'].replace('0', 'none')

# Initialize lists to store graph data
all_nodes = []
all_edges = []
all_graphs = []

# Process each row in training_data to create a graph
for graph_id in range(len(training_data)):
    row = training_data.iloc[graph_id]

    # Create nodes for 14 standard layers and 1 top layer
    for layer in range(15):  # Layers 0-13 are standard, 14 is top
        for _, node in nodes_data.iterrows():
            node_id = node['node_id']
            wall_attr = node['wall']
            glass_attr = node['glass']
            if layer < 14:  # Standard layer: wall, glass, ceiling
                ceiling_attr = node['ceiling']  # Note: 'celling' is typo in nodes.xlsx
                wall_value = 0.0 if wall_attr == 'none' else row[attribute_mapping[wall_attr]]
                glass_value = 0.0 if glass_attr == 'none' else row[attribute_mapping[glass_attr]]
                ceiling_value = 0.0 if ceiling_attr == 'none' else row[attribute_mapping[ceiling_attr]]
                all_nodes.append({
                    'graph_id': graph_id,
                    'node_id': node_id,
                    'layer': layer,
                    'wall': wall_value,
                    'glass': glass_value,
                    'ceiling': ceiling_value
                })
            else:  # Top layer: wall, glass, roof (using XPS_roof for roof)
                roof_value = row['XPS_roof']  # Always use XPS_roof for roof
                wall_value = 0.0 if wall_attr == 'none' else row[attribute_mapping[wall_attr]]
                glass_value = 0.0 if glass_attr == 'none' else row[attribute_mapping[glass_attr]]
                all_nodes.append({
                    'graph_id': graph_id,
                    'node_id': node_id,
                    'layer': layer,
                    'wall': wall_value,
                    'glass': glass_value,
                    'ceiling': roof_value
                })

    # Create intra-layer edges for each layer
    for layer in range(15):
        for _, edge in edges_data.iterrows():
            all_edges.append({
                'graph_id': graph_id,
                'source': edge['source'],
                'target': edge['target'],
                'layer_source': layer,
                'layer_target': layer,
                'adjacency': edge['adjacency'],
                'connection': edge['connection'],
                'stack': 0
            })

    # Create inter-layer edges (between identical nodes in adjacent layers)
    for layer in range(14):  # Connect layers 0-1, 1-2, ..., 13-14
        for node_id in range(19):  # Nodes 0-18
            all_edges.append({
                'graph_id': graph_id,
                'source': node_id,
                'target': node_id,
                'layer_source': layer,
                'layer_target': layer + 1,
                'adjacency': 0,
                'connection': 0,
                'stack': 1
            })

    # Store graph-level data
    all_graphs.append({
        'graph_id': graph_id,
        'OC': row['OC']
    })

# Convert to DataFrames
nodes_df = pd.DataFrame(all_nodes)
edges_df = pd.DataFrame(all_edges)
graphs_df = pd.DataFrame(all_graphs)

# Save to Excel with multiple sheets
with pd.ExcelWriter('building_graphs_optimized.xlsx') as writer:
    nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
    edges_df.to_excel(writer, sheet_name='Edges', index=False)
    graphs_df.to_excel(writer, sheet_name='GraphData', index=False)

print(
    "Optimized graphs with 'none' mapping (compatible with '0' in nodes.xlsx) have been successfully created and saved to 'building_graphs_optimized.xlsx'.")