# GNNvsANNvsXGBvsGPR_FULL.py
# 完整四模型对比 + 图规模扩展性实验
# 作者：你的名字
# 日期：2025

import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 模型定义 ====================
class EdgeAttrGAT(torch.nn.Module):
    def __init__(self, node_features=5, edge_features=3, hidden_channels=128, heads=8):
        super().__init__()
        self.conv1 = GATConv(node_features, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, edge_dim=edge_features)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, edge_dim=edge_features)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.3)
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=2.0)
        if self.lin.bias is not None:
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.nn.functional.elu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.conv3(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.conv4(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        return self.lin(x).squeeze(-1)

class ANNModel(torch.nn.Module):
    def __init__(self, input_features, hidden_channels=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_features, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.3)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=2.0)
        if self.fc4.bias is not None:
            torch.nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = torch.nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x).squeeze(-1)

# ==================== 数据加载与预处理 ====================
def load_data():
    try:
        nodes_df = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='Nodes')
        edges_df = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='Edges')
        graphs_df = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='GraphData')
        logger.info(f"Loaded {len(graphs_df)} graphs, Total range: {graphs_df['Total'].min():.2f} ~ {graphs_df['Total'].max():.2f}")
        return nodes_df, edges_df, graphs_df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess_data(nodes_df, edges_df, graphs_df):
    node_scaler = StandardScaler()
    glass_encoder = OneHotEncoder(sparse_output=False, categories=[[1, 2, 3]], handle_unknown='ignore')
    target_scaler = StandardScaler()

    node_features = node_scaler.fit_transform(nodes_df[['wall', 'ceiling']])
    glass_encoded = glass_encoder.fit_transform(nodes_df[['glass']])
    node_features = np.concatenate([node_features, glass_encoded], axis=1)

    edge_features = edges_df[['adjacency', 'connection', 'stack']].values
    target_values = target_scaler.fit_transform(graphs_df[['Total']]).flatten()

    # GNN 数据
    gnn_data_list = []
    for idx, graph_id in enumerate(graphs_df['graph_id']):
        graph_nodes = nodes_df[nodes_df['graph_id'] == graph_id]
        graph_edges = edges_df[edges_df['graph_id'] == graph_id]
        if len(graph_nodes) != 285 or len(graph_edges) != 701:
            logger.warning(f"Graph {graph_id} has {len(graph_nodes)} nodes, skipping")
            continue
        x = torch.tensor(node_features[graph_nodes.index], dtype=torch.float)
        edge_index = torch.tensor(graph_edges[['source', 'target']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_features[graph_edges.index], dtype=torch.float)
        y = torch.tensor(target_values[idx], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, graph_id=graph_id)
        gnn_data_list.append(data)

    # ANN 聚合特征
    ann_features = []
    for graph_id in graphs_df['graph_id']:
        nodes = nodes_df[nodes_df['graph_id'] == graph_id]
        wall_stats = nodes['wall'].agg(['mean', 'std', 'min', 'max'])
        ceiling_stats = nodes['ceiling'].agg(['mean', 'std', 'min', 'max'])
        glass_counts = nodes['glass'].value_counts(normalize=True).reindex([1, 2, 3], fill_value=0)
        features = np.concatenate([wall_stats.values, ceiling_stats.values, glass_counts.values])
        ann_features.append(features)

    ann_scaler = StandardScaler()
    ann_features_scaled = ann_scaler.fit_transform(ann_features)
    ann_data_list = [(torch.tensor(f, dtype=torch.float), torch.tensor(t, dtype=torch.float))
                     for f, t in zip(ann_features_scaled, target_values)]

    logger.info(f"Processed {len(gnn_data_list)} graphs for GNN, {len(ann_data_list)} for ANN/XGB/GPR")
    return gnn_data_list, ann_data_list, node_scaler, glass_encoder, target_scaler, ann_scaler

# ==================== 四模型训练与对比 ====================
def train_and_compare_all_models(gnn_data_list, ann_data_list, node_scaler, glass_encoder, target_scaler, ann_scaler, device):
    logger.info("Starting 4-model comparison...")

    # 统一划分（保持随机性一致）
    train_idx, temp_idx = train_test_split(range(len(gnn_data_list)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.6667, random_state=42)

    train_gnn = [gnn_data_list[i] for i in train_idx]
    val_gnn   = [gnn_data_list[i] for i in val_idx]
    test_gnn  = [gnn_data_list[i] for i in test_idx]

    train_ann = [ann_data_list[i] for i in train_idx]
    val_ann   = [ann_data_list[i] for i in val_idx]
    test_ann  = [ann_data_list[i] for i in test_idx]

    # ANN/XGB/GPR 使用的特征和标签（原始尺度）
    X_train = np.stack([x.numpy() for x, y in train_ann])
    y_train = target_scaler.inverse_transform(np.array([y.item() for x, y in train_ann]).reshape(-1, 1)).flatten()
    X_test  = np.stack([x.numpy() for x, y in test_ann])
    y_test  = target_scaler.inverse_transform(np.array([y.item() for x, y in test_ann]).reshape(-1, 1)).flatten()

    # 关键：准备标准化空间的真实标签（只算一次）
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    results = {}
    predictions = {}
    predictions_scaled = {}   # 额外保存标准化空间的预测，用于后续分析（可选）

    # ==================== 1. GNN ====================
    logger.info("Training GNN...")
    start_time = time.time()
    gnn_model = EdgeAttrGAT().to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_gnn + [d.clone() for d in train_gnn for _ in range(2)], batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_gnn, batch_size=16, shuffle=False)

    # 训练（你原来只跑1轮，这里建议改成200+，否则GNN没学到东西）
    for epoch in range(500):
        gnn_model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gnn_model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # 测试时同时收集原始和标准化预测
    gnn_model.eval()
    gnn_preds_raw = []
    gnn_preds_scaled = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out_scaled = gnn_model(batch)                    # 标准化空间输出
            out_raw = target_scaler.inverse_transform(out_scaled.cpu().numpy().reshape(-1, 1)).flatten()
            gnn_preds_raw.extend(out_raw)
            gnn_preds_scaled.extend(out_scaled.cpu().numpy().flatten())

    gnn_time = time.time() - start_time
    results['GNN'] = {
        'MSE': mean_squared_error(y_test, gnn_preds_raw),
        'MAE': mean_absolute_error(y_test, gnn_preds_raw),
        'R2': r2_score(y_test, gnn_preds_raw),
        'MSE_scaled': mean_squared_error(y_test_scaled, gnn_preds_scaled),
        'MAE_scaled': mean_absolute_error(y_test_scaled, gnn_preds_scaled),
        'Time': gnn_time
    }
    predictions['GNN'] = np.array(gnn_preds_raw)
    predictions_scaled['GNN'] = np.array(gnn_preds_scaled)
    torch.save(gnn_model.state_dict(), 'best_gnn.pth')

    # ==================== 2. ANN ====================
    logger.info("Training ANN...")
    start_time = time.time()
    ann_model = ANNModel(input_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(ann_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = torch.nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float).to(device)
    y_train_scaled_t = torch.tensor(target_scaler.transform(y_train.reshape(-1, 1)).flatten(), dtype=torch.float).to(device)

    for epoch in range(500):  # ANN 可以多训一点
        ann_model.train()
        optimizer.zero_grad()
        out_scaled = ann_model(X_train_t)
        loss = criterion(out_scaled, y_train_scaled_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    ann_model.eval()
    with torch.no_grad():
        ann_pred_scaled = ann_model(torch.tensor(X_test, dtype=torch.float).to(device)).cpu().numpy().flatten()
        ann_pred_raw = target_scaler.inverse_transform(ann_pred_scaled.reshape(-1, 1)).flatten()

    ann_time = time.time() - start_time
    results['ANN'] = {
        'MSE': mean_squared_error(y_test, ann_pred_raw),
        'MAE': mean_absolute_error(y_test, ann_pred_raw),
        'R2': r2_score(y_test, ann_pred_raw),
        'MSE_scaled': mean_squared_error(y_test_scaled, ann_pred_scaled),
        'MAE_scaled': mean_absolute_error(y_test_scaled, ann_pred_scaled),
        'Time': ann_time
    }
    predictions['ANN'] = ann_pred_raw
    predictions_scaled['ANN'] = ann_pred_scaled
    torch.save(ann_model.state_dict(), 'best_ann.pth')

    # ==================== 3. XGBoost ====================
    logger.info("Training XGBoost (native API, fully stable)...")
    start_time = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    evals_result = {}
    xgb_model = xgb.train(
        params={
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'nthread': -1,
        },
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )

    best_iter = xgb_model.best_iteration
    logger.info(f"XGBoost early stopped at iteration {best_iter}")

    xgb_pred_raw = xgb_model.predict(dtest, iteration_range=(0, best_iter))
    xgb_pred_scaled = target_scaler.transform(xgb_pred_raw.reshape(-1, 1)).flatten()

    xgb_time = time.time() - start_time
    results['XGBoost'] = {
        'MSE': mean_squared_error(y_test, xgb_pred_raw),
        'MAE': mean_absolute_error(y_test, xgb_pred_raw),
        'R2': r2_score(y_test, xgb_pred_raw),
        'MSE_scaled': mean_squared_error(y_test_scaled, xgb_pred_scaled),
        'MAE_scaled': mean_absolute_error(y_test_scaled, xgb_pred_scaled),
        'Time': xgb_time,
        'Best_Iteration': best_iter
    }
    predictions['XGBoost'] = xgb_pred_raw
    predictions_scaled['XGBoost'] = xgb_pred_scaled
    xgb_model.save_model('best_xgb.json')

    # ==================== 4. GPR ====================
    logger.info("Training GPR...")
    start_time = time.time()
    kernel = C(1.0) * RBF(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, random_state=42)
    gpr.fit(X_train, y_train)
    gpr_pred_raw = gpr.predict(X_test)
    gpr_pred_scaled = target_scaler.transform(gpr_pred_raw.reshape(-1, 1)).flatten()
    gpr_time = time.time() - start_time

    results['GPR'] = {
        'MSE': mean_squared_error(y_test, gpr_pred_raw),
        'MAE': mean_absolute_error(y_test, gpr_pred_raw),
        'R2': r2_score(y_test, gpr_pred_raw),
        'MSE_scaled': mean_squared_error(y_test_scaled, gpr_pred_scaled),
        'MAE_scaled': mean_absolute_error(y_test_scaled, gpr_pred_scaled),
        'Time': gpr_time
    }
    predictions['GPR'] = gpr_pred_raw
    predictions_scaled['GPR'] = gpr_pred_scaled
    joblib.dump(gpr, 'best_gpr.pkl')

    # ==================== 结果汇总与保存 ====================
    pd.DataFrame(results).T.to_csv('model_comparison_summary.csv')

    # 最终美观表格（同时显示原始和标准化指标）
    compare_df = pd.DataFrame({
        'Model':      list(results.keys()),
        'MSE':        [f"{v['MSE']:,.2f}" for v in results.values()],
        'MSE_scaled': [f"{v['MSE_scaled']:.6f}" for v in results.values()],
        'MAE':        [f"{v['MAE']:.2f}" for v in results.values()],
        'MAE_scaled': [f"{v['MAE_scaled']:.6f}" for v in results.values()],
        'R²':         [f"{v['R2']:.4f}" for v in results.values()],
        'Time(s)':    [f"{v['Time']:.2f}" for v in results.values()],
    })
    if 'Best_Iteration' in results['XGBoost']:
        compare_df['Best_Iter'] = ['—'] * len(compare_df)
        compare_df.loc[compare_df['Model'] == 'XGBoost', 'Best_Iter'] = results['XGBoost']['Best_Iteration']

    logger.info("\n" + "="*80)
    logger.info("FINAL PERFORMANCE COMPARISON (Raw + Scaled Metrics)")
    logger.info("="*80)
    logger.info(compare_df.to_string(index=False))
    logger.info("="*80)

    compare_df.to_csv('model_comparison_final_with_scaled.csv', index=False)
    logger.info("Detailed comparison (with scaled metrics) saved: model_comparison_final_with_scaled.csv")

    # 误差分布图（仍然用原始尺度，更直观）
    plt.figure(figsize=(12, 6))
    for name, pred in predictions.items():
        errors = np.abs(y_test - pred)
        sns.kdeplot(errors, label=f'{name} (MAE={mean_absolute_error(y_test, pred):.2f})', linewidth=2)
    plt.xlabel('Absolute Error (t CO₂e)')
    plt.ylabel('Density')
    plt.title('Absolute Error Distribution Comparison (Raw Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution_4models.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 随机样本尺度检查（确认都在7000~8000左右）
    sample_indices = np.random.choice(len(y_test), size=3, replace=False)
    check_df = pd.DataFrame({'True_CO2e': y_test[sample_indices].round(2)})
    for name, pred in predictions.items():
        check_df[name] = pred[sample_indices].round(2)
    logger.info("\nScale consistency check (3 random samples):")
    logger.info(check_df.to_string(index=False))
    check_df.to_csv('prediction_scale_check_samples.csv', index=False)

    return results, predictions

# ==================== 图规模扩展性实验 ====================
# ==================== 图规模扩展性实验（修复版）===================
def scalability_test(node_scaler, glass_encoder, target_scaler, ann_scaler, device):
    logger.info("Starting Scalability Test (50/100/200/285 nodes)...")
    nodes_df = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='Nodes')
    edges_df = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='Edges')
    sample_id = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='GraphData')['graph_id'].iloc[0]
    sample_nodes = nodes_df[nodes_df['graph_id'] == sample_id].copy()
    sample_edges = edges_df[edges_df['graph_id'] == sample_id].copy()
    true_y = pd.read_excel('building_graphs_optimized.xlsx', sheet_name='GraphData').iloc[0]['Total']

    scales = [50, 100, 200, 285]
    results = []

    # 加载训练好的模型
    gnn_model = EdgeAttrGAT().to(device)
    gnn_model.load_state_dict(torch.load('best_gnn.pth', map_location=device))
    gnn_model.eval()

    ann_model = ANNModel(input_features=ann_scaler.n_features_in_).to(device)
    ann_model.load_state_dict(torch.load('best_ann.pth', map_location=device))
    ann_model.eval()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('best_xgb.json')

    gpr = joblib.load('best_gpr.pkl')

    for n in scales:
        row = {'Nodes': n}
        nodes_per_layer = 19
        layers = max(1, n // nodes_per_layer)
        keep = layers * nodes_per_layer if n < 285 else 285
        node_idx = sample_nodes.index[:keep]
        node_map = {old: new for new, old in enumerate(node_idx)}

        sub_nodes = sample_nodes.loc[node_idx].reset_index(drop=True)
        sub_edges = sample_edges[sample_edges['source'].isin(node_idx) & sample_edges['target'].isin(node_idx)].copy()
        sub_edges['source'] = sub_edges['source'].map(node_map)
        sub_edges['target'] = sub_edges['target'].map(node_map)

        # GNN 输入
        x_raw = node_scaler.transform(sub_nodes[['wall', 'ceiling']])
        glass_enc = glass_encoder.transform(sub_nodes[['glass']])
        x = torch.tensor(np.hstack([x_raw, glass_enc]), dtype=torch.float).to(device)
        edge_index = torch.tensor(sub_edges[['source', 'target']].values.T, dtype=torch.long).to(device)
        edge_attr = torch.tensor(sub_edges[['adjacency', 'connection', 'stack']].values, dtype=torch.float).to(device)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        # ANN 输入
        wall_stats = sub_nodes['wall'].agg(['mean', 'std', 'min', 'max'])
        ceil_stats = sub_nodes['ceiling'].agg(['mean', 'std', 'min', 'max'])
        glass_prop = sub_nodes['glass'].value_counts(normalize=True).reindex([1,2,3], fill_value=0)
        ann_feat = np.hstack([wall_stats.values, ceil_stats.values, glass_prop.values])
        ann_input = torch.tensor(ann_scaler.transform(ann_feat.reshape(1, -1)), dtype=torch.float).to(device)

        # ========== GNN 推理时间 & 内存（兼容CPU）==========
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = gnn_model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        gnn_inf_time = (time.time() - start_time) / 20

        memory_mb = 0
        if device.type == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        # 修复预测部分
        with torch.no_grad():
            gnn_output = gnn_model(data)
            pred_gnn = target_scaler.inverse_transform(
                gnn_output.detach().cpu().numpy().reshape(-1, 1)
            )[0, 0]

        row.update({
            'GNN_Time_ms': gnn_inf_time * 1000,
            'GNN_Memory_MB': memory_mb,
            'GNN_MAE': abs(pred_gnn - true_y)
        })

        # ========== ANN ==========
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = ann_model(ann_input)
        row['ANN_Time_ms'] = (time.time() - start_time) / 100 * 1000

        # ========== XGBoost ==========
        x_input = ann_input.cpu().numpy()
        start_time = time.time()
        for _ in range(100):
            _ = xgb_model.predict(x_input)
        row['XGBoost_Time_ms'] = (time.time() - start_time) / 100 * 1000

        # ========== GPR ==========
        start_time = time.time()
        for _ in range(10):
            _ = gpr.predict(x_input)
        row['GPR_Time_ms'] = (time.time() - start_time) / 10 * 1000

        results.append(row)
        logger.info(f"Scale {n} nodes completed.")

    df = pd.DataFrame(results)
    df.to_csv('scalability_comparison.csv', index=False)
    # ... 绘图部分不变 ...

    # 绘图
    plt.figure(figsize=(10,6))
    plt.plot(df['Nodes'], df['GNN_Time_ms'], 'o-', label='GNN', markersize=8)
    plt.plot(df['Nodes'], df['ANN_Time_ms'], 's-', label='ANN', markersize=8)
    plt.plot(df['Nodes'], df['XGBoost_Time_ms'], '^-', label='XGBoost', markersize=8)
    plt.plot(df['Nodes'], df['GPR_Time_ms'], 'd-', label='GPR', markersize=8)
    plt.yscale('log')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Inference Time (ms)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('scalability_inference_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Scalability test completed!")




# ==================== 主程序 ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    nodes_df, edges_df, graphs_df = load_data()
    gnn_data_list, ann_data_list, node_scaler, glass_encoder, target_scaler, ann_scaler = preprocess_data(nodes_df, edges_df, graphs_df)

    # 第一步：训练四个模型 + 性能对比
    results, predictions = train_and_compare_all_models(
        gnn_data_list, ann_data_list,
        node_scaler, glass_encoder, target_scaler, ann_scaler, device
    )

    # 第二步：图规模扩展性实验
    scalability_test(node_scaler, glass_encoder, target_scaler, ann_scaler, device)

    logger.info("=== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ===")
    logger.info("Generated files: model_comparison_summary.csv, error_distribution_4models.png, scalability_comparison.csv, scalability_inference_time.png")

