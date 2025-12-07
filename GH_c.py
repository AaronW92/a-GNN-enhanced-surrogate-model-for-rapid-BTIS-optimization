# -*- coding: utf-8 -*-
"""
Python Script
Created on  Monday August 2025 04:48:14
@author:  IT Support

[desc]
Description of the plugin Here
Write here any thing...
[/desc]

ARGUMENTS:
----------
<inp>
    _input1 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input2 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input3 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input4 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input5 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input6 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input7 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input8 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point
</inp>
<inp>
    _input9 :[required] - [type = string] - [default = None]
    Descripe your input here
        * bullet point.
        * bullet point

RETURN:
----------
    <out>
        output_ : indicate your output description here. \n refers to a new line.
    </out>

"""
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import logging
import os

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

a = float(_input1/200)
b = float(_input2/200)
c = float(_input3/200)
d = float(_input4/100)
e = float(_input5/200)
f = float(_input6/200)
g = float(_input7)
h = float(_input8)
i = float(_input9)

# 定义 EdgeAttrGAT 模型（保持不变）
class EdgeAttrGAT(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels=128, heads=8):
        super(EdgeAttrGAT, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(node_features, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv2 = torch_geometric.nn.GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv3 = torch_geometric.nn.GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv4 = torch_geometric.nn.GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, edge_dim=edge_features)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.3)
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=2.0)
        if self.lin.bias is not None:
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = torch.nn.functional.elu(x)
        x = self.dropout(x)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = self.lin(x).squeeze(-1)
        return x

# 输入值验证和转换函数
def validate_and_convert_input(value, param_name, expected_type, allowed_values=None):
    try:
        # 如果输入是字符串，尝试去除多余的引号并转换为目标类型
        if isinstance(value, str):
            value = value.strip("'\"")  # 去除可能的单引号或双引号
            value = float(value) if expected_type == float else int(value)
        # 确保值是目标类型
        if expected_type == float:
            value = float(value)
        elif expected_type == int:
            value = int(value)
        # 检查允许的值范围（如果提供）
        if allowed_values:
            if isinstance(allowed_values, tuple):
                min_val, max_val = allowed_values
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Parameter {param_name}={value} out of range [{min_val}, {max_val}]")
            else:
                if value not in allowed_values:
                    raise ValueError(f"Parameter {param_name}={value} not in allowed values {allowed_values}")
        return value
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input for {param_name}: {value}, error: {e}")
        raise ValueError(f"Invalid input for {param_name}: {value}, expected {expected_type.__name__}")

# 预测函数（稍作调整以支持输入验证）
def predict_energy(params, model, node_scaler, glass_encoder, target_scaler, device):
    try:
        logger.info("Starting energy prediction...")
        # 使用绝对路径加载文件
        nodes_data = pd.read_excel(r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\nodes.xlsx')
        edges_data = pd.read_excel(r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\edges.xlsx')

        expected_nodes = 19
        if len(nodes_data) != expected_nodes:
            raise ValueError(f"nodes.xlsx contains {len(nodes_data)} nodes, expected {expected_nodes}")
        expected_columns = ['node_id', 'wall', 'glass', 'ceiling', 'roof']
        if not all(col in nodes_data.columns for col in expected_columns):
            raise ValueError(f"nodes.xlsx missing required columns: {expected_columns}")

        expected_edges_per_layer = 29
        if len(edges_data) != expected_edges_per_layer:
            logger.warning(f"edges.xlsx contains {len(edges_data)} edges, expected {expected_edges_per_layer}")

        # 参数范围定义
        param_ranges = {
            'PUR': (0.05, 0.2),
            'XPS_roof': (0.05, 0.2),
            'XPS_floor': (0.03, 0.1),
            'Mortar': (0.01, 0.04),
            'Silicene_wall': (0.05, 0.2),
            'Silicene_floor': (0.03, 0.1),
            'BW_Low-e': [1, 2, 3],
            'EW_Low-e': [1, 2, 3],
            'CW_Low-e': [1, 2, 3]
        }

        # 验证和转换输入参数
        validated_params = {}
        for key, value in params.items():
            expected_type = float if key in ['PUR', 'XPS_roof', 'XPS_floor', 'Mortar', 'Silicene_wall', 'Silicene_floor'] else int
            validated_params[key] = validate_and_convert_input(value, key, expected_type, param_ranges.get(key))

        # 属性映射
        attribute_mapping = {
            'A': 'PUR', 'B': 'XPS_roof', 'C': 'XPS_floor', 'D': 'Mortar',
            'E': 'Silicene_wall', 'F': 'Silicene_floor', 'G': 'BW_Low-e',
            'H': 'EW_Low-e', 'I': 'CW_Low-e', 'none': 0.0
        }

        # 构建节点特征
        node_features = []
        for layer in range(15):
            for _, node in nodes_data.iterrows():
                wall = validated_params.get(attribute_mapping.get(node['wall'], 'none'), 0.0)
                ceiling = validated_params.get(attribute_mapping.get(node['ceiling'], 'none'), 0.0) if layer < 14 else validated_params.get('XPS_roof', 0.0)
                glass = validated_params.get(attribute_mapping.get(node['glass'], 'none'), 0.0)
                node_features.append([wall, ceiling, glass])

        node_features_df = pd.DataFrame(node_features, columns=['wall', 'ceiling', 'glass'])
        scaled_features = node_scaler.transform(node_features_df[['wall', 'ceiling']])
        glass_values = node_features_df[['glass']]
        glass_encoded = glass_encoder.transform(glass_values)
        node_features = np.concatenate([scaled_features, glass_encoded], axis=1)

        # 构建边索引和边属性
        edge_indices = []
        edge_attrs = []
        for layer in range(15):
            offset = layer * 19
            for _, edge in edges_data.iterrows():
                edge_indices.append([edge['source'] + offset, edge['target'] + offset])
                edge_attrs.append([edge['adjacency'], edge['connection'], edge.get('stack', 0)])

        for layer in range(14):
            for node_id in range(19):
                edge_indices.append([node_id + layer * 19, node_id + (layer + 1) * 19])
                edge_attrs.append([0, 0, 1])

        expected_total_edges = 29 * 15 + 14 * 19
        if len(edge_indices) != expected_total_edges:
            logger.warning(f"Generated {len(edge_indices)} edges, expected {expected_total_edges}")

        # 转换为张量
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # 构建图数据
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            pred = model(data)
            pred = target_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))

        logger.info(f"Predicted embodied carbon: {pred.item():.4f}")
        return pred.item()

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

# 主程序
if __name__ == "__main__":
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # 加载预训练模型和预处理器
        model = EdgeAttrGAT(node_features=5, edge_features=3, hidden_channels=128, heads=8).to(device)

        model_path = r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\best_gnn.pth'
        raw_dict = torch.load(model_path, map_location=device)

        # 关键就这一行！！！把摊平的参数重新包装成标准 state_dict
        state_dict = {k: v for k, v in raw_dict.items()}  # 其实已经就是 state_dict，只是没被 OrderedDict 包裹

        model.load_state_dict(state_dict)
        model.eval()
        logger.info("模型加载成功！（已兼容打散参数保存方式）")

        node_scaler = joblib.load(r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\node_scaler.joblib')
        glass_encoder = joblib.load(r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\glass_encoder.joblib')
        target_scaler = joblib.load(r'D:\20250628 carbon emission and energy efficiency\graphgeneration\pythonProject\target_scaler.joblib')

        # Grasshopper 输入参数
        params = {
            'PUR': a,
            'XPS_roof': b,
            'XPS_floor': c,
            'Mortar': d,
            'Silicene_wall': e,
            'Silicene_floor': f,
            'BW_Low-e': g,
            'EW_Low-e': h,
            'CW_Low-e': i
        }

        # 调用预测函数
        output_ = predict_energy(params, model, node_scaler, glass_encoder, target_scaler, device)
        logger.info(f"Final predicted embodied carbon: {output_:.4f}")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise