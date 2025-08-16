#!/usr/bin/env python3
"""
双头模型训练脚本 - 专注于E.coli和S.aureus
基于共享骨干网络 + 两个独立输出头的架构
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入数据预处理和特征工程模块
sys.path.append('/root/NKU-TMU_AMP_project')
from data_preprocessing import GRAMPAPreprocessor
from feature_engineering import AMP_FeatureExtractor

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class DualHeadDataset(Dataset):
    """双头模型数据集类"""
    def __init__(self, features_dict, bacteria_mapping, scaler=None, fit_scaler=False):
        """
        初始化数据集
        
        Args:
            features_dict: 特征字典
            bacteria_mapping: 菌株映射 {'escherichia_coli': 0, 'staphylococcus_aureus': 1}
            scaler: 标准化器
            fit_scaler: 是否拟合标准化器
        """
        self.plm_embeddings = features_dict['plm_embeddings']
        self.physicochemical_features = features_dict['physicochemical_features']
        self.targets = features_dict['targets']
        self.bacteria_names = features_dict['bacteria_names']
        self.sequences = features_dict['sequences']
        
        # 可选字段
        self.is_censored = features_dict.get('is_censored', None)
        self.censoring_threshold = features_dict.get('censoring_threshold', None)
        self.sample_weights = features_dict.get('sample_weights', None)
        
        # 菌株ID映射
        self.bacteria_mapping = bacteria_mapping
        self.bacteria_ids = np.array([bacteria_mapping[name] for name in self.bacteria_names])
        
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
        
        # 统计菌株分布
        bacteria_counts = pd.Series(self.bacteria_names).value_counts()
        print("菌株分布:")
        for bacteria, count in bacteria_counts.items():
            print(f"  {bacteria}: {count} ({100*count/len(self.bacteria_names):.1f}%)")
        
        if self.is_censored is not None:
            censored_count = np.sum(self.is_censored)
            print(f"删失样本数: {censored_count} / {len(self.is_censored)} ({100*censored_count/len(self.is_censored):.1f}%)")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        item = {
            'features': torch.FloatTensor(self.combined_features[idx]),
            'target': torch.FloatTensor([self.targets[idx]]),
            'bacteria_id': torch.LongTensor([self.bacteria_ids[idx]]),
            'bacteria_name': self.bacteria_names[idx],
            'sequence': self.sequences[idx]
        }
        
        if self.is_censored is not None:
            item['is_censored'] = torch.BoolTensor([self.is_censored[idx]])
            item['censoring_threshold'] = torch.FloatTensor([self.censoring_threshold[idx]])
        
        if self.sample_weights is not None:
            item['sample_weight'] = torch.FloatTensor([self.sample_weights[idx]])
        
        return item

class DualHeadModel(nn.Module):
    """双头模型：共享骨干 + 两个独立输出头"""
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        
        # 共享骨干网络
        backbone_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # Swish激活函数
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 两个独立的输出头
        self.head_ecoli = nn.Linear(prev_dim, 1)      # Head₀: E.coli
        self.head_saureus = nn.Linear(prev_dim, 1)    # Head₁: S.aureus
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, features):
        # 共享骨干
        shared_features = self.backbone(features)
        
        # 两个独立输出头
        pred_ecoli = self.head_ecoli(shared_features).squeeze(-1)
        pred_saureus = self.head_saureus(shared_features).squeeze(-1)
        
        return pred_ecoli, pred_saureus

class DualHeadLoss(nn.Module):
    """双头模型的损失函数（删失感知 + 稳健）"""
    def __init__(self, delta=1.0, censoring_weight=1.0):
        super().__init__()
        self.delta = delta
        self.censoring_weight = censoring_weight
    
    def huber_loss(self, pred, target):
        """Huber损失"""
        abs_error = torch.abs(pred - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + self.delta * linear
    
    def forward(self, pred_ecoli, pred_saureus, targets, bacteria_ids, 
                is_censored=None, censoring_threshold=None):
        """
        计算双头损失
        
        Args:
            pred_ecoli: E.coli预测值 (batch_size,)
            pred_saureus: S.aureus预测值 (batch_size,)
            targets: 真实值 (batch_size,)
            bacteria_ids: 菌株ID (batch_size,) - 0: E.coli, 1: S.aureus
            is_censored: 删失标记 (batch_size,)
            censoring_threshold: 删失阈值 (batch_size,)
        """
        batch_size = len(targets)
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            bacteria_id = bacteria_ids[i].item()
            target = targets[i]
            
            # 根据菌株ID选择对应的预测头
            if bacteria_id == 0:  # E.coli
                pred = pred_ecoli[i]
            else:  # S.aureus
                pred = pred_saureus[i]
            
            # 计算该样本的损失
            if is_censored is not None and is_censored[i]:
                # 删失样本：右删失约束
                threshold = censoring_threshold[i]
                censored_loss = torch.clamp(threshold - pred, min=0) ** 2
                sample_loss = self.censoring_weight * censored_loss
            else:
                # 非删失样本：Huber损失
                sample_loss = self.huber_loss(pred, target)
            
            total_loss += sample_loss
            valid_samples += 1
        
        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0)

class DualHeadTrainer:
    """双头模型训练器"""
    def __init__(self, device='auto', output_dir='model_outputs'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"使用设备: {self.device}")
        
        # 菌株映射
        self.bacteria_mapping = {
            'escherichia_coli': 0,
            'staphylococcus_aureus': 1
        }
        self.id_to_bacteria = {0: 'escherichia_coli', 1: 'staphylococcus_aureus'}
    
    def prepare_data(self, input_file='/root/NKU-TMU_AMP_project/data/AMP/grampa_merged_dataset.csv'):
        """准备双头模型数据"""
        print("=== 步骤1: 数据预处理（仅保留E.coli和S.aureus）===")
        
        # 使用数据预处理器
        preprocessor = GRAMPAPreprocessor(
            input_file=input_file,
            output_dir='/root/NKU-TMU_AMP_project/processed_data'
        )
        
        # 运行预处理流程
        preprocessor.run_full_pipeline()
        
        # 加载聚合数据
        aggregated_df = pd.read_csv('/root/NKU-TMU_AMP_project/processed_data/grampa_aggregated_full.csv')
        
        # 过滤只保留两个目标菌株
        target_bacteria = ['escherichia_coli', 'staphylococcus_aureus']
        filtered_df = aggregated_df[aggregated_df['bacterium'].isin(target_bacteria)].copy()
        
        print(f"原始聚合数据: {len(aggregated_df)} 样本")
        print(f"过滤后数据: {len(filtered_df)} 样本")
        
        # 统计过滤后的菌株分布
        bacteria_counts = filtered_df['bacterium'].value_counts()
        print("过滤后菌株分布:")
        for bacteria, count in bacteria_counts.items():
            print(f"  {bacteria}: {count}")
        
        # 保存过滤后的数据集
        for split in ['train', 'val', 'test']:
            split_df = filtered_df[filtered_df['split'] == split].copy()
            if len(split_df) > 0:
                output_file = f'/root/NKU-TMU_AMP_project/processed_data/grampa_conditional_{split}_top2.csv'
                split_df.to_csv(output_file, index=False)
                print(f"保存 {split} 集: {output_file} ({len(split_df)} 样本)")
        
        return filtered_df
    
    def extract_features(self):
        """提取特征"""
        print("\n=== 步骤2: 特征提取 ===")
        
        # 使用特征提取器
        extractor = AMP_FeatureExtractor(
            processed_data_dir='/root/NKU-TMU_AMP_project/processed_data',
            features_output_dir='/root/NKU-TMU_AMP_project/features',
            device=self.device
        )
        
        # 为双头数据集提取特征
        datasets = {}
        for split in ['train', 'val', 'test']:
            file_path = f'/root/NKU-TMU_AMP_project/processed_data/grampa_conditional_{split}_top2.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # 收集唯一序列
                sequences = df['sequence'].unique()
                
                # 提取PLM embeddings
                plm_embeddings = extractor.extract_plm_embeddings(sequences.tolist())
                
                # 提取理化特征
                physchem_features = extractor.calculate_physicochemical_features(sequences.tolist())
                
                # 标准化理化特征
                if split == 'train':
                    scaler = StandardScaler()
                    physchem_features_scaled = scaler.fit_transform(physchem_features)
                    self.scaler = scaler
                else:
                    physchem_features_scaled = self.scaler.transform(physchem_features)
                
                # 创建序列到特征的映射
                seq_to_plm = {seq: plm_embeddings[i] for i, seq in enumerate(sequences)}
                seq_to_physchem = {seq: physchem_features_scaled[i] for i, seq in enumerate(sequences)}
                
                # 为数据集中的每个样本分配特征
                dataset_plm = np.array([seq_to_plm[seq] for seq in df['sequence']])
                dataset_physchem = np.array([seq_to_physchem[seq] for seq in df['sequence']])
                
                # 构建特征字典
                features_dict = {
                    'plm_embeddings': dataset_plm,
                    'physicochemical_features': dataset_physchem,
                    'sequences': df['sequence'].values,
                    'bacteria_names': df['bacterium'].values,
                    'targets': df['value_winsorized'].values,
                    'is_censored': df['is_censored'].values,
                    'censoring_threshold': df['censoring_threshold'].fillna(0).values,
                    'sample_weights': df['n_measurements'].values
                }
                
                # 保存特征文件
                output_file = f'/root/NKU-TMU_AMP_project/features/dual_head_{split}_features.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(features_dict, f)
                
                datasets[split] = features_dict
                print(f"{split}集特征提取完成: PLM {dataset_plm.shape}, 理化 {dataset_physchem.shape}")
        
        return datasets
    
    def create_datasets(self, features_data):
        """创建PyTorch数据集"""
        print("\n=== 步骤3: 创建PyTorch数据集 ===")
        
        datasets = {}
        
        # 训练集
        train_dataset = DualHeadDataset(
            features_data['train'], 
            self.bacteria_mapping,
            fit_scaler=True
        )
        datasets['train'] = train_dataset
        self.scaler = train_dataset.scaler
        
        # 验证集和测试集
        for split in ['val', 'test']:
            if split in features_data:
                dataset = DualHeadDataset(
                    features_data[split],
                    self.bacteria_mapping,
                    scaler=self.scaler
                )
                datasets[split] = dataset
        
        return datasets
    
    def train_model(self, datasets, batch_size=128, learning_rate=1e-4, 
                   num_epochs=100, patience=15):
        """训练双头模型"""
        print("\n=== 步骤4: 训练双头模型 ===")
        
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        
        # 创建平衡采样器
        bacteria_ids = train_dataset.bacteria_ids
        bacteria_counts = np.bincount(bacteria_ids)
        bacteria_weights = 1.0 / np.sqrt(bacteria_counts + 1e-8)
        sample_weights = bacteria_weights[bacteria_ids]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        input_dim = train_dataset.combined_features.shape[1]
        model = DualHeadModel(input_dim=input_dim, hidden_dims=[512, 256], dropout=0.2).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        criterion = DualHeadLoss(delta=1.0, censoring_weight=1.0)
        
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
                bacteria_ids = batch['bacteria_id'].squeeze(-1).to(self.device)
                is_censored = batch['is_censored'].squeeze(-1).to(self.device)
                censoring_threshold = batch['censoring_threshold'].squeeze(-1).to(self.device)
                
                optimizer.zero_grad()
                pred_ecoli, pred_saureus = model(features)
                loss = criterion(pred_ecoli, pred_saureus, targets, bacteria_ids, 
                               is_censored, censoring_threshold)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_predictions = {0: [], 1: []}  # 分菌株记录预测
            val_targets = {0: [], 1: []}
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    targets = batch['target'].squeeze(-1).to(self.device)
                    bacteria_ids = batch['bacteria_id'].squeeze(-1).to(self.device)
                    is_censored = batch['is_censored'].squeeze(-1).to(self.device)
                    censoring_threshold = batch['censoring_threshold'].squeeze(-1).to(self.device)
                    
                    pred_ecoli, pred_saureus = model(features)
                    loss = criterion(pred_ecoli, pred_saureus, targets, bacteria_ids,
                                   is_censored, censoring_threshold)
                    val_loss += loss.item()
                    
                    # 分菌株记录预测结果
                    for i in range(len(targets)):
                        bacteria_id = bacteria_ids[i].item()
                        target = targets[i].item()
                        
                        if bacteria_id == 0:
                            pred = pred_ecoli[i].item()
                        else:
                            pred = pred_saureus[i].item()
                        
                        val_predictions[bacteria_id].append(pred)
                        val_targets[bacteria_id].append(target)
            
            val_loss /= len(val_loader)
            
            # 计算分菌株评估指标
            val_metrics = {}
            for bacteria_id in [0, 1]:
                if len(val_targets[bacteria_id]) > 0:
                    preds = np.array(val_predictions[bacteria_id])
                    targs = np.array(val_targets[bacteria_id])
                    rmse = np.sqrt(mean_squared_error(targs, preds))
                    r2 = r2_score(targs, preds)
                    val_metrics[self.id_to_bacteria[bacteria_id]] = {'rmse': rmse, 'r2': r2}
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            for bacteria_name, metrics in val_metrics.items():
                print(f"  {bacteria_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(self.output_dir, 'dual_head_best.pt'))
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
        
        with open(os.path.join(self.output_dir, 'dual_head_training_history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(self.output_dir, 'dual_head_best.pt')))
        self.model = model
        
        return model, history
    
    def evaluate_model(self, datasets):
        """评估模型"""
        print("\n=== 步骤5: 模型评估 ===")
        
        test_dataset = datasets['test']
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        self.model.eval()
        predictions = {0: [], 1: []}
        targets = {0: [], 1: []}
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                batch_targets = batch['target'].squeeze(-1).to(self.device)
                bacteria_ids = batch['bacteria_id'].squeeze(-1).to(self.device)
                
                pred_ecoli, pred_saureus = self.model(features)
                
                # 分菌株记录结果
                for i in range(len(batch_targets)):
                    bacteria_id = bacteria_ids[i].item()
                    target = batch_targets[i].item()
                    
                    if bacteria_id == 0:
                        pred = pred_ecoli[i].item()
                    else:
                        pred = pred_saureus[i].item()
                    
                    predictions[bacteria_id].append(pred)
                    targets[bacteria_id].append(target)
        
        # 计算评估指标
        results = {}
        for bacteria_id in [0, 1]:
            bacteria_name = self.id_to_bacteria[bacteria_id]
            preds = np.array(predictions[bacteria_id])
            targs = np.array(targets[bacteria_id])
            
            if len(targs) > 0:
                rmse = np.sqrt(mean_squared_error(targs, preds))
                r2 = r2_score(targs, preds)
                
                # 计算二分类指标
                thresholds = {'2μM': np.log10(2), '5μM': np.log10(5), '10μM': np.log10(10)}
                binary_metrics = {}
                
                for name, threshold in thresholds.items():
                    y_true = (targs <= threshold).astype(int)
                    y_scores = -preds  # 负号：越小的预测值越可能是活性的
                    
                    if len(np.unique(y_true)) == 2:
                        auc = roc_auc_score(y_true, y_scores)
                        ap = average_precision_score(y_true, y_scores)
                        binary_metrics[name] = {'auc': auc, 'ap': ap}
                
                results[bacteria_name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(targs),
                    'predictions': preds,
                    'targets': targs,
                    'binary_metrics': binary_metrics
                }
                
                print(f"{bacteria_name} 测试结果:")
                print(f"  样本数: {len(targs)}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R²: {r2:.4f}")
                
                for name, metrics in binary_metrics.items():
                    print(f"  {name} - AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
        
        # 保存结果
        with open(os.path.join(self.output_dir, 'dual_head_test_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def plot_results(self, history, test_results):
        """绘制结果图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练曲线
        axes[0, 0].plot(history['train_losses'], label='训练损失', alpha=0.7)
        axes[0, 0].plot(history['val_losses'], label='验证损失', alpha=0.7)
        axes[0, 0].set_title('双头模型训练曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 预测vs真实值散点图
        for i, (bacteria_name, results) in enumerate(test_results.items()):
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            preds = results['predictions']
            targs = results['targets']
            
            axes[row, col].scatter(targs, preds, alpha=0.6)
            axes[row, col].plot([targs.min(), targs.max()], [targs.min(), targs.max()], 'r--', alpha=0.8)
            axes[row, col].set_xlabel('真实值 (log MIC)')
            axes[row, col].set_ylabel('预测值 (log MIC)')
            axes[row, col].set_title(f'{bacteria_name}\nR²={results["r2"]:.4f}, RMSE={results["rmse"]:.4f}')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dual_head_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"结果图已保存: {os.path.join(self.output_dir, 'dual_head_results.png')}")
    
    def run_full_pipeline(self, input_file='/root/NKU-TMU_AMP_project/data/AMP/grampa_merged_dataset.csv'):
        """运行完整流程"""
        print("开始双头模型训练流程...")
        
        # 步骤1: 数据准备
        self.prepare_data(input_file)
        
        # 步骤2: 特征提取
        features_data = self.extract_features()
        
        # 步骤3: 创建数据集
        datasets = self.create_datasets(features_data)
        
        # 步骤4: 训练模型
        model, history = self.train_model(datasets)
        
        # 步骤5: 评估模型
        test_results = self.evaluate_model(datasets)
        
        # 步骤6: 绘制结果
        self.plot_results(history, test_results)
        
        # 生成报告
        self.generate_report(history, test_results)
        
        print(f"\n双头模型训练完成！结果保存在: {self.output_dir}")
        
        return model, history, test_results
    
    def generate_report(self, history, test_results):
        """生成训练报告"""
        report_file = os.path.join(self.output_dir, 'dual_head_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 双头模型训练报告\n\n")
            
            f.write("## 模型架构\n")
            f.write("- 共享骨干网络: 512 → 256\n")
            f.write("- 两个独立输出头: E.coli 和 S.aureus\n")
            f.write("- 激活函数: SiLU\n")
            f.write("- 正则化: LayerNorm + Dropout(0.2)\n\n")
            
            f.write("## 训练结果\n")
            f.write(f"- 训练轮数: {len(history['train_losses'])}\n")
            f.write(f"- 最佳验证损失: {history['best_val_loss']:.4f}\n")
            f.write(f"- 最终训练损失: {history['train_losses'][-1]:.4f}\n")
            f.write(f"- 最终验证损失: {history['val_losses'][-1]:.4f}\n\n")
            
            f.write("## 测试结果\n")
            for bacteria_name, results in test_results.items():
                f.write(f"### {bacteria_name}\n")
                f.write(f"- 样本数: {results['n_samples']}\n")
                f.write(f"- RMSE: {results['rmse']:.4f}\n")
                f.write(f"- R²: {results['r2']:.4f}\n")
                
                f.write("- 二分类指标:\n")
                for threshold, metrics in results['binary_metrics'].items():
                    f.write(f"  - {threshold}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}\n")
                f.write("\n")
        
        print(f"训练报告已保存: {report_file}")

def main():
    """主函数"""
    # 创建训练器
    trainer = DualHeadTrainer(
        device='auto',
        output_dir='/root/NKU-TMU_AMP_project/model_outputs'
    )
    
    # 运行完整流程
    model, history, test_results = trainer.run_full_pipeline()

if __name__ == "__main__":
    main()
