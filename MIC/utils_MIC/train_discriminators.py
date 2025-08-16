#!/usr/bin/env python3
"""
两层建模方案训练脚本
基于筛选器设计.md的第四部分要求

实现：
A. 序列聚合回归（~7k样本，总体活性预测）
B. 条件回归（~45k样本，序列×菌株细粒度预测）
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class AMP_Dataset(Dataset):
    """AMP数据集类"""
    def __init__(self, features_dict, scaler=None, fit_scaler=False):
        """
        初始化数据集
        
        Args:
            features_dict: 特征字典
            scaler: 标准化器
            fit_scaler: 是否拟合标准化器
        """
        self.plm_embeddings = features_dict['plm_embeddings']
        self.physicochemical_features = features_dict['physicochemical_features']
        self.targets = features_dict['targets']
        self.sequences = features_dict['sequences']
        
        # 可选字段
        self.bacteria_ids = features_dict.get('bacteria_ids', None)
        self.bacteria_names = features_dict.get('bacteria_names', None)
        self.is_censored = features_dict.get('is_censored', None)
        self.censoring_threshold = features_dict.get('censoring_threshold', None)
        self.sample_weights = features_dict.get('sample_weights', None)
        
        # 标准化理化特征
        if scaler is None:
            self.scaler = StandardScaler()
            fit_scaler = True
        else:
            self.scaler = scaler
            
        if fit_scaler:
            self.physicochemical_features_scaled = self.scaler.fit_transform(self.physicochemical_features)
        else:
            self.physicochemical_features_scaled = self.scaler.transform(self.physicochemical_features)
        
        # 合并特征
        self.combined_features = np.concatenate([
            self.plm_embeddings,
            self.physicochemical_features_scaled
        ], axis=1)
        
        print(f"数据集大小: {len(self.targets)}")
        print(f"特征维度: PLM={self.plm_embeddings.shape[1]}, 理化={self.physicochemical_features.shape[1]}")
        print(f"合并特征维度: {self.combined_features.shape[1]}")
        
        if self.bacteria_ids is not None:
            print(f"包含菌株信息: {len(np.unique(self.bacteria_ids))} 个唯一菌株")
        if self.is_censored is not None:
            censored_count = np.sum(self.is_censored)
            print(f"删失样本数: {censored_count} / {len(self.is_censored)} ({100*censored_count/len(self.is_censored):.1f}%)")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {
            'features': torch.FloatTensor(self.combined_features[idx]),
            'target': torch.FloatTensor([self.targets[idx]]),
            'sequence': self.sequences[idx]
        }
        
        if self.bacteria_ids is not None:
            item['bacteria_id'] = torch.LongTensor([self.bacteria_ids[idx]])
        
        if self.is_censored is not None:
            item['is_censored'] = torch.BoolTensor([self.is_censored[idx]])
            item['censoring_threshold'] = torch.FloatTensor([self.censoring_threshold[idx]])
        
        if self.sample_weights is not None:
            item['sample_weight'] = torch.FloatTensor([self.sample_weights[idx]])
        
        return item

class SequenceRegressionModel(nn.Module):
    """序列聚合回归模型（模型A）"""
    def __init__(self, input_dim, hidden_dims=[1024, 512], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # Swish激活函数
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, features):
        return self.model(features).squeeze(-1)

class ConditionalRegressionModel(nn.Module):
    """条件回归模型（模型B）"""
    def __init__(self, input_dim, n_bacteria, bacteria_embedding_dim=32, 
                 hidden_dims=[1024, 512], dropout=0.3):
        super().__init__()
        
        # 菌株embedding层
        self.bacteria_embedding = nn.Embedding(n_bacteria, bacteria_embedding_dim)
        
        # 合并输入维度
        combined_input_dim = input_dim + bacteria_embedding_dim
        
        layers = []
        prev_dim = combined_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, features, bacteria_ids):
        # 获取菌株embedding
        bacteria_emb = self.bacteria_embedding(bacteria_ids.squeeze(-1))
        
        # 拼接特征
        combined = torch.cat([features, bacteria_emb], dim=-1)
        
        return self.model(combined).squeeze(-1)

class HuberLoss(nn.Module):
    """Huber损失函数"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic**2 + self.delta * linear)

class CensoringAwareHuberLoss(nn.Module):
    """删失感知的Huber损失"""
    def __init__(self, delta=1.0, censoring_weight=2.0):
        super().__init__()
        self.delta = delta
        self.censoring_weight = censoring_weight
        self.huber = HuberLoss(delta)
    
    def forward(self, pred, target, is_censored=None, censoring_threshold=None):
        if is_censored is None:
            # 没有删失信息，使用标准Huber损失
            return self.huber(pred, target)
        
        # 非删失样本的损失
        uncensored_mask = ~is_censored
        uncensored_loss = 0.0
        if uncensored_mask.sum() > 0:
            uncensored_loss = self.huber(pred[uncensored_mask], target[uncensored_mask])
        
        # 删失样本的损失（右删失）
        censored_mask = is_censored
        censored_loss = 0.0
        if censored_mask.sum() > 0:
            # 只有当预测值小于删失阈值时才惩罚
            censored_pred = pred[censored_mask]
            censored_thresh = censoring_threshold[censored_mask]
            violation = torch.clamp(censored_thresh - censored_pred, min=0)
            censored_loss = torch.mean(violation ** 2) * self.censoring_weight
        
        # 加权组合
        n_uncensored = uncensored_mask.sum().float()
        n_censored = censored_mask.sum().float()
        n_total = len(pred)
        
        if n_uncensored > 0 and n_censored > 0:
            total_loss = (n_uncensored / n_total) * uncensored_loss + (n_censored / n_total) * censored_loss
        elif n_uncensored > 0:
            total_loss = uncensored_loss
        else:
            total_loss = censored_loss
        
        return total_loss

class AMPTrainer:
    """AMP模型训练器"""
    def __init__(self, device='auto', output_dir='model_outputs'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
    
    def load_data(self, features_dir='features'):
        """加载特征数据"""
        print("正在加载特征数据...")
        
        self.datasets = {}
        self.scalers = {}
        
        # 加载序列聚合数据集
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(features_dir, f'sequence_{split}_features.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    features_dict = pickle.load(f)
                
                # 第一个数据集拟合scaler
                if f'sequence_{split}' not in self.scalers:
                    if split == 'train':
                        dataset = AMP_Dataset(features_dict, fit_scaler=True)
                        self.scalers['sequence'] = dataset.scaler
                    else:
                        dataset = AMP_Dataset(features_dict, scaler=self.scalers['sequence'])
                else:
                    dataset = AMP_Dataset(features_dict, scaler=self.scalers['sequence'])
                
                self.datasets[f'sequence_{split}'] = dataset
                print(f"加载序列数据集 {split}: {len(dataset)} 样本")
        
        # 加载条件回归数据集
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(features_dir, f'conditional_{split}_features.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    features_dict = pickle.load(f)
                
                if f'conditional_{split}' not in self.scalers:
                    if split == 'train':
                        dataset = AMP_Dataset(features_dict, fit_scaler=True)
                        self.scalers['conditional'] = dataset.scaler
                    else:
                        dataset = AMP_Dataset(features_dict, scaler=self.scalers['conditional'])
                else:
                    dataset = AMP_Dataset(features_dict, scaler=self.scalers['conditional'])
                
                self.datasets[f'conditional_{split}'] = dataset
                print(f"加载条件数据集 {split}: {len(dataset)} 样本")
        
        # 加载菌株映射
        bacteria_mapping_file = os.path.join(features_dir, 'bacteria_mapping.pkl')
        with open(bacteria_mapping_file, 'rb') as f:
            self.bacteria_mapping = pickle.load(f)
        
        print(f"菌株数量: {len(self.bacteria_mapping['bacteria_to_id'])}")
    
    def train_sequence_model(self, batch_size=128, learning_rate=2e-4, 
                           num_epochs=100, patience=10):
        """训练序列聚合回归模型"""
        print("\n=== 训练序列聚合回归模型 ===")
        
        train_dataset = self.datasets['sequence_train']
        val_dataset = self.datasets['sequence_val']
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_dim = train_dataset.combined_features.shape[1]
        model = SequenceRegressionModel(input_dim=input_dim).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        criterion = HuberLoss(delta=1.0)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                features = batch['features'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)
                
                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].squeeze(-1).to(self.device)
                    
                    predictions = model(features)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # 计算评估指标
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            val_r2 = r2_score(val_targets, val_predictions)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val RMSE={val_rmse:.4f}, Val R²={val_r2:.4f}")
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(self.output_dir, 'sequence_regression_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(self.output_dir, 'sequence_training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(os.path.join(self.output_dir, 'sequence_regression_best.pt')))
        self.sequence_model = model
        
        # 在测试集上评估
        self.evaluate_sequence_model()
        
        return model, history
    
    def train_conditional_model(self, batch_size=128, learning_rate=2e-4,
                              num_epochs=100, patience=10):
        """训练条件回归模型"""
        print("\n=== 训练条件回归模型 ===")
        
        train_dataset = self.datasets['conditional_train']
        val_dataset = self.datasets['conditional_val']
        
        # 创建加权采样器（平衡菌株分布）
        bacteria_counts = np.bincount(train_dataset.bacteria_ids.flatten())
        bacteria_weights = 1.0 / np.sqrt(bacteria_counts + 1e-8)  # 避免除零
        sample_weights = bacteria_weights[train_dataset.bacteria_ids.flatten()]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_dim = train_dataset.combined_features.shape[1]
        n_bacteria = len(self.bacteria_mapping['bacteria_to_id'])
        model = ConditionalRegressionModel(
            input_dim=input_dim,
            n_bacteria=n_bacteria,
            bacteria_embedding_dim=32
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        criterion = CensoringAwareHuberLoss(delta=1.0, censoring_weight=2.0)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                features = batch['features'].to(self.device)
                bacteria_ids = batch['bacteria_id'].to(self.device)
                targets = batch['target'].squeeze(-1).to(self.device)
                is_censored = batch['is_censored'].squeeze(-1).to(self.device)
                censoring_threshold = batch['censoring_threshold'].squeeze(-1).to(self.device)
                
                optimizer.zero_grad()
                predictions = model(features, bacteria_ids)
                loss = criterion(predictions, targets, is_censored, censoring_threshold)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    bacteria_ids = batch['bacteria_id'].to(self.device)
                    targets = batch['target'].squeeze(-1).to(self.device)
                    is_censored = batch['is_censored'].squeeze(-1).to(self.device)
                    censoring_threshold = batch['censoring_threshold'].squeeze(-1).to(self.device)
                    
                    predictions = model(features, bacteria_ids)
                    loss = criterion(predictions, targets, is_censored, censoring_threshold)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # 计算评估指标
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
            val_r2 = r2_score(val_targets, val_predictions)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val RMSE={val_rmse:.4f}, Val R²={val_r2:.4f}")
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(self.output_dir, 'conditional_regression_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        with open(os.path.join(self.output_dir, 'conditional_training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(os.path.join(self.output_dir, 'conditional_regression_best.pt')))
        self.conditional_model = model
        
        # 在测试集上评估
        self.evaluate_conditional_model()
        
        return model, history
    
    def evaluate_sequence_model(self):
        """评估序列聚合模型"""
        print("\n=== 评估序列聚合模型 ===")
        
        test_dataset = self.datasets['sequence_test']
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        self.sequence_model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                batch_targets = batch['target'].squeeze(-1).to(self.device)
                
                batch_predictions = self.sequence_model(features)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        # 计算回归指标
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        print(f"测试集 RMSE: {rmse:.4f}")
        print(f"测试集 R²: {r2:.4f}")
        
        # 计算二分类指标（不同阈值）
        thresholds = {
            '2μM': np.log10(2),    # ≈ 0.301
            '5μM': np.log10(5),    # ≈ 0.699  
            '10μM': np.log10(10)   # = 1.0
        }
        
        for name, threshold in thresholds.items():
            y_true = (np.array(targets) <= threshold).astype(int)
            y_scores = -np.array(predictions)  # 负号：越小的预测值越可能是活性的
            
            if len(np.unique(y_true)) == 2:  # 确保有两个类别
                auc = roc_auc_score(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
                print(f"{name} 阈值 - AUC: {auc:.4f}, AP: {ap:.4f}")
        
        # 保存预测结果
        results = {
            'predictions': predictions,
            'targets': targets,
            'rmse': rmse,
            'r2': r2
        }
        
        with open(os.path.join(self.output_dir, 'sequence_test_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def evaluate_conditional_model(self):
        """评估条件回归模型"""
        print("\n=== 评估条件回归模型 ===")
        
        test_dataset = self.datasets['conditional_test']
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        self.conditional_model.eval()
        predictions = []
        targets = []
        bacteria_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                batch_bacteria_ids = batch['bacteria_id'].to(self.device)
                batch_targets = batch['target'].squeeze(-1).to(self.device)
                
                batch_predictions = self.conditional_model(features, batch_bacteria_ids)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                bacteria_ids.extend(batch_bacteria_ids.squeeze(-1).cpu().numpy())
        
        # 整体指标
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        print(f"整体测试集 RMSE: {rmse:.4f}")
        print(f"整体测试集 R²: {r2:.4f}")
        
        # 按菌株评估
        unique_bacteria = np.unique(bacteria_ids)
        bacteria_results = {}
        
        for bacteria_id in unique_bacteria:
            mask = np.array(bacteria_ids) == bacteria_id
            if mask.sum() >= 10:  # 至少10个样本
                bacteria_pred = np.array(predictions)[mask]
                bacteria_target = np.array(targets)[mask]
                
                bacteria_rmse = np.sqrt(mean_squared_error(bacteria_target, bacteria_pred))
                bacteria_r2 = r2_score(bacteria_target, bacteria_pred)
                
                bacteria_name = self.bacteria_mapping['id_to_bacteria'][bacteria_id]
                bacteria_results[bacteria_name] = {
                    'rmse': bacteria_rmse,
                    'r2': bacteria_r2,
                    'n_samples': mask.sum()
                }
        
        # 显示Top 10菌株结果
        print("\nTop 10 菌株结果:")
        sorted_bacteria = sorted(bacteria_results.items(), 
                               key=lambda x: x[1]['n_samples'], reverse=True)
        
        for bacteria_name, results in sorted_bacteria[:10]:
            print(f"{bacteria_name}: RMSE={results['rmse']:.4f}, "
                  f"R²={results['r2']:.4f}, N={results['n_samples']}")
        
        # 计算二分类指标
        thresholds = {
            '2μM': np.log10(2),
            '5μM': np.log10(5),
            '10μM': np.log10(10)
        }
        
        for name, threshold in thresholds.items():
            y_true = (np.array(targets) <= threshold).astype(int)
            y_scores = -np.array(predictions)
            
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
                print(f"{name} 阈值 - AUC: {auc:.4f}, AP: {ap:.4f}")
        
        # 保存结果
        results = {
            'predictions': predictions,
            'targets': targets,
            'bacteria_ids': bacteria_ids,
            'rmse': rmse,
            'r2': r2,
            'bacteria_results': bacteria_results
        }
        
        with open(os.path.join(self.output_dir, 'conditional_test_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 序列模型训练曲线
        if os.path.exists(os.path.join(self.output_dir, 'sequence_training_history.pkl')):
            with open(os.path.join(self.output_dir, 'sequence_training_history.pkl'), 'rb') as f:
                seq_history = pickle.load(f)
            
            axes[0].plot(seq_history['train_losses'], label='训练损失', alpha=0.7)
            axes[0].plot(seq_history['val_losses'], label='验证损失', alpha=0.7)
            axes[0].set_title('序列聚合模型训练曲线')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 条件模型训练曲线
        if os.path.exists(os.path.join(self.output_dir, 'conditional_training_history.pkl')):
            with open(os.path.join(self.output_dir, 'conditional_training_history.pkl'), 'rb') as f:
                cond_history = pickle.load(f)
            
            axes[1].plot(cond_history['train_losses'], label='训练损失', alpha=0.7)
            axes[1].plot(cond_history['val_losses'], label='验证损失', alpha=0.7)
            axes[1].set_title('条件回归模型训练曲线')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 创建训练器
    trainer = AMPTrainer(device='auto', output_dir='/root/NKU-TMU_AMP_project/model_outputs')
    
    # 加载数据
    trainer.load_data(features_dir='/root/NKU-TMU_AMP_project/features')
    
    # 训练序列聚合模型
    print("\n" + "="*50)
    print("开始训练序列聚合回归模型")
    print("="*50)
    
    sequence_model, seq_history = trainer.train_sequence_model(
        batch_size=128,
        learning_rate=2e-4,
        num_epochs=100,
        patience=10
    )
    
    # 训练条件回归模型
    print("\n" + "="*50)
    print("开始训练条件回归模型")
    print("="*50)
    
    conditional_model, cond_history = trainer.train_conditional_model(
        batch_size=128,
        learning_rate=2e-4,
        num_epochs=100,
        patience=10
    )
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    print("\n" + "="*50)
    print("训练完成！模型和结果已保存到 model_outputs/ 目录")
    print("="*50)

if __name__ == "__main__":
    main()
