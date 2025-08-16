#!/usr/bin/env python3
"""
特征构建脚本
实现PLM表征、理化特征和菌株表示的生成
基于筛选器设计.md的第三部分要求
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 理化特征计算
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math

class AMP_FeatureExtractor:
    def __init__(self, 
                 processed_data_dir='processed_data',
                 features_output_dir='features',
                 esm_model_name='facebook/esm2_t33_650M_UR50D',
                 device='auto'):
        """
        初始化特征提取器
        
        Args:
            processed_data_dir: 处理后数据目录
            features_output_dir: 特征输出目录
            esm_model_name: ESM模型名称
            device: 计算设备
        """
        self.processed_data_dir = processed_data_dir
        self.features_output_dir = features_output_dir
        self.esm_model_name = esm_model_name
        
        # 设备配置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(features_output_dir, exist_ok=True)
        
        # 初始化ESM模型和tokenizer
        print("正在加载ESM-2模型...")
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        self.esm_model = EsmModel.from_pretrained(esm_model_name).to(self.device)
        self.esm_model.eval()
        print("ESM-2模型加载完成")
        
        # 氨基酸属性字典
        self.aa_properties = {
            # 疏水性 (Kyte-Doolittle scale)
            'hydrophobicity': {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            },
            # 电荷 (pH 7.4)
            'charge': {
                'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
                'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
                'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
            },
            # 极性
            'polarity': {
                'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
                'Q': 1, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
                'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
                'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
            }
        }
    
    def extract_plm_embeddings(self, sequences, batch_size=32, max_length=512):
        """
        提取PLM (ESM-2) 表征
        
        Args:
            sequences: 序列列表
            batch_size: 批次大小
            max_length: 最大序列长度
            
        Returns:
            embeddings: shape (n_sequences, 2*hidden_dim) 的embedding矩阵
        """
        print(f"正在提取 {len(sequences)} 个序列的PLM embeddings...")
        
        all_embeddings = []
        
        # 批次处理
        for i in tqdm(range(0, len(sequences), batch_size), desc="提取PLM embeddings"):
            batch_sequences = sequences[i:i+batch_size]
            
            # 过滤掉过长的序列
            valid_sequences = []
            valid_indices = []
            for j, seq in enumerate(batch_sequences):
                if len(seq) <= max_length:
                    valid_sequences.append(seq)
                    valid_indices.append(j)
            
            if not valid_sequences:
                # 如果批次中没有有效序列，添加零向量
                batch_embeddings = torch.zeros(len(batch_sequences), 2 * self.esm_model.config.hidden_size)
                all_embeddings.append(batch_embeddings)
                continue
            
            # Tokenize
            inputs = self.tokenizer(
                valid_sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
                # 获取序列表征 (去除CLS和SEP tokens)
                sequence_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # (batch, seq_len, hidden_dim)
                
                # 池化操作
                mean_pooled = torch.mean(sequence_embeddings, dim=1)  # (batch, hidden_dim)
                max_pooled = torch.max(sequence_embeddings, dim=1)[0]  # (batch, hidden_dim)
                
                # 拼接均值和最大池化
                combined_embeddings = torch.cat([mean_pooled, max_pooled], dim=1)  # (batch, 2*hidden_dim)
                
                # 创建完整批次的embedding矩阵
                batch_embeddings = torch.zeros(len(batch_sequences), 2 * self.esm_model.config.hidden_size)
                for j, valid_idx in enumerate(valid_indices):
                    batch_embeddings[valid_idx] = combined_embeddings[j].cpu()
                
                all_embeddings.append(batch_embeddings)
        
        # 合并所有批次
        final_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"PLM embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings.numpy()
    
    def calculate_physicochemical_features(self, sequences):
        """
        计算理化特征
        
        Args:
            sequences: 序列列表
            
        Returns:
            features: 理化特征矩阵
        """
        print(f"正在计算 {len(sequences)} 个序列的理化特征...")
        
        features = []
        
        for seq in tqdm(sequences, desc="计算理化特征"):
            seq_features = {}
            
            # 处理无效序列
            if not seq or pd.isna(seq) or len(seq) == 0:
                # 返回零特征
                features.append([0] * 30)  # 预计30个特征
                continue
            
            # 清理序列（移除非标准氨基酸）
            clean_seq = ''.join([aa for aa in seq.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])
            if len(clean_seq) == 0:
                features.append([0] * 30)
                continue
            
            try:
                # 使用BioPython计算基本特征
                analysis = ProteinAnalysis(clean_seq)
                
                # 1. 长度
                length = len(clean_seq)
                
                # 2. 分子量
                mw = analysis.molecular_weight()
                
                # 3. 等电点
                try:
                    isoelectric_point = analysis.isoelectric_point()
                except:
                    isoelectric_point = 7.0
                
                # 4. GRAVY (疏水性)
                try:
                    gravy = analysis.gravy()
                except:
                    gravy = 0.0
                
                # 5. 净电荷 (pH 7.4)
                net_charge = sum([self.aa_properties['charge'].get(aa, 0) for aa in clean_seq])
                
                # 6. RK含量 (碱性残基比例)
                rk_count = clean_seq.count('R') + clean_seq.count('K')
                rk_ratio = rk_count / length if length > 0 else 0
                
                # 7. 氨基酸组成 (20维)
                aa_composition = []
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    aa_composition.append(clean_seq.count(aa) / length if length > 0 else 0)
                
                # 8. 疏水矩 (假设α螺旋)
                hydrophobic_moment = self.calculate_hydrophobic_moment(clean_seq)
                
                # 9. 其他特征
                positive_charge = sum([1 for aa in clean_seq if self.aa_properties['charge'].get(aa, 0) > 0])
                negative_charge = sum([1 for aa in clean_seq if self.aa_properties['charge'].get(aa, 0) < 0])
                polar_residues = sum([1 for aa in clean_seq if self.aa_properties['polarity'].get(aa, 0) == 1])
                
                # 组装特征向量
                feature_vector = [
                    length,
                    mw,
                    isoelectric_point,
                    gravy,
                    net_charge,
                    rk_ratio,
                    hydrophobic_moment,
                    positive_charge / length if length > 0 else 0,
                    negative_charge / length if length > 0 else 0,
                    polar_residues / length if length > 0 else 0
                ] + aa_composition
                
                features.append(feature_vector)
                
            except Exception as e:
                print(f"计算序列 {seq[:20]}... 的理化特征时出错: {e}")
                features.append([0] * 30)
        
        features_array = np.array(features, dtype=np.float32)
        print(f"理化特征 shape: {features_array.shape}")
        
        return features_array
    
    def calculate_hydrophobic_moment(self, sequence, window=100):
        """
        计算疏水矩 (假设α螺旋结构)
        
        Args:
            sequence: 氨基酸序列
            window: 螺旋角度窗口 (度)
            
        Returns:
            hydrophobic_moment: 疏水矩值
        """
        if len(sequence) == 0:
            return 0.0
        
        # α螺旋每个残基旋转100度
        angle_per_residue = math.radians(window)
        
        sum_x = 0
        sum_y = 0
        
        for i, aa in enumerate(sequence):
            hydrophobicity = self.aa_properties['hydrophobicity'].get(aa, 0)
            angle = i * angle_per_residue
            
            sum_x += hydrophobicity * math.cos(angle)
            sum_y += hydrophobicity * math.sin(angle)
        
        hydrophobic_moment = math.sqrt(sum_x**2 + sum_y**2) / len(sequence)
        return hydrophobic_moment
    
    def create_bacteria_embeddings(self, bacteria_names, embedding_dim=32):
        """
        创建菌株embedding映射
        
        Args:
            bacteria_names: 菌株名称列表
            embedding_dim: embedding维度
            
        Returns:
            bacteria_to_id: 菌株名到ID的映射
            embedding_matrix: embedding矩阵
        """
        print(f"正在创建菌株embeddings...")
        
        # 统计菌株频次
        bacteria_counts = pd.Series(bacteria_names).value_counts()
        print(f"唯一菌株数: {len(bacteria_counts)}")
        
        # 创建菌株到ID的映射
        bacteria_to_id = {}
        id_to_bacteria = {}
        
        # 为常见菌株分配ID
        for i, (bacteria, count) in enumerate(bacteria_counts.items()):
            bacteria_to_id[bacteria] = i
            id_to_bacteria[i] = bacteria
        
        # 创建可学习的embedding矩阵
        n_bacteria = len(bacteria_to_id)
        embedding_matrix = np.random.normal(0, 0.1, (n_bacteria, embedding_dim)).astype(np.float32)
        
        print(f"菌株embedding shape: {embedding_matrix.shape}")
        
        return bacteria_to_id, embedding_matrix, id_to_bacteria
    
    def process_all_datasets(self):
        """
        处理所有数据集，生成特征
        """
        print("开始特征工程流程...")
        
        # 1. 加载处理后的数据
        datasets = {}
        for split in ['train', 'val', 'test']:
            # 条件回归数据集 (包含菌株信息)
            conditional_file = os.path.join(self.processed_data_dir, f'grampa_conditional_{split}.csv')
            if os.path.exists(conditional_file):
                try:
                    datasets[f'conditional_{split}'] = pd.read_csv(conditional_file, low_memory=False)
                    print(f"加载 {conditional_file}: {len(datasets[f'conditional_{split}'])} 样本")
                except Exception as e:
                    print(f"加载 {conditional_file} 失败: {e}")
                    continue
            
            # 序列回归数据集 (不含菌株信息)
            sequence_file = os.path.join(self.processed_data_dir, f'grampa_sequence_{split}.csv')
            if os.path.exists(sequence_file):
                try:
                    datasets[f'sequence_{split}'] = pd.read_csv(sequence_file, low_memory=False)
                    print(f"加载 {sequence_file}: {len(datasets[f'sequence_{split}'])} 样本")
                except Exception as e:
                    print(f"加载 {sequence_file} 失败: {e}")
                    continue
        
        # 2. 收集所有唯一序列
        all_sequences = set()
        for dataset_name, df in datasets.items():
            if 'sequence' in df.columns:
                all_sequences.update(df['sequence'].dropna().unique())
        
        all_sequences = sorted(list(all_sequences))
        print(f"总共 {len(all_sequences)} 个唯一序列")
        
        # 3. 提取PLM embeddings
        print("\n=== 提取PLM embeddings ===")
        plm_embeddings = self.extract_plm_embeddings(all_sequences)
        
        # 创建序列到embedding的映射
        seq_to_embedding = {seq: plm_embeddings[i] for i, seq in enumerate(all_sequences)}
        
        # 4. 计算理化特征
        print("\n=== 计算理化特征 ===")
        physicochemical_features = self.calculate_physicochemical_features(all_sequences)
        
        # 创建序列到理化特征的映射
        seq_to_physchem = {seq: physicochemical_features[i] for i, seq in enumerate(all_sequences)}
        
        # 5. 处理菌株embeddings (仅针对条件回归数据集)
        print("\n=== 创建菌株embeddings ===")
        all_bacteria = set()
        for dataset_name, df in datasets.items():
            if 'conditional' in dataset_name and 'bacterium' in df.columns:
                all_bacteria.update(df['bacterium'].dropna().unique())
        
        all_bacteria = sorted(list(all_bacteria))
        bacteria_to_id, bacteria_embedding_matrix, id_to_bacteria = self.create_bacteria_embeddings(all_bacteria)
        
        # 6. 为每个数据集生成特征
        print("\n=== 为各数据集生成特征 ===")
        for dataset_name, df in datasets.items():
            print(f"\n处理数据集: {dataset_name}")
            
            # 序列特征
            sequences = df['sequence'].values
            dataset_plm_embeddings = np.array([seq_to_embedding.get(seq, np.zeros(plm_embeddings.shape[1])) 
                                             for seq in sequences])
            dataset_physchem_features = np.array([seq_to_physchem.get(seq, np.zeros(physicochemical_features.shape[1])) 
                                                for seq in sequences])
            
            # 保存序列特征
            features_dict = {
                'plm_embeddings': dataset_plm_embeddings,
                'physicochemical_features': dataset_physchem_features,
                'sequences': sequences
            }
            
            # 如果是条件回归数据集，添加菌株信息
            if 'conditional' in dataset_name and 'bacterium' in df.columns:
                bacteria = df['bacterium'].values
                bacteria_ids = np.array([bacteria_to_id.get(bact, 0) for bact in bacteria])
                features_dict['bacteria_ids'] = bacteria_ids
                features_dict['bacteria_names'] = bacteria
            
            # 添加目标变量和其他信息
            if 'value_winsorized' in df.columns:
                features_dict['targets'] = df['value_winsorized'].values
            if 'is_censored' in df.columns:
                features_dict['is_censored'] = df['is_censored'].values
            if 'censoring_threshold' in df.columns:
                features_dict['censoring_threshold'] = df['censoring_threshold'].fillna(0).values
            if 'n_measurements' in df.columns:
                features_dict['sample_weights'] = df['n_measurements'].values
            
            # 保存特征文件
            output_file = os.path.join(self.features_output_dir, f'{dataset_name}_features.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(features_dict, f)
            print(f"保存特征文件: {output_file}")
            print(f"  - PLM embeddings: {dataset_plm_embeddings.shape}")
            print(f"  - 理化特征: {dataset_physchem_features.shape}")
            if 'bacteria_ids' in features_dict:
                print(f"  - 菌株IDs: {features_dict['bacteria_ids'].shape}")
        
        # 7. 保存全局映射和embedding矩阵
        print("\n=== 保存全局映射 ===")
        
        # 保存序列特征映射
        seq_features_mapping = {
            'seq_to_plm': seq_to_embedding,
            'seq_to_physchem': seq_to_physchem,
            'plm_embedding_dim': plm_embeddings.shape[1],
            'physchem_feature_dim': physicochemical_features.shape[1]
        }
        
        with open(os.path.join(self.features_output_dir, 'sequence_features_mapping.pkl'), 'wb') as f:
            pickle.dump(seq_features_mapping, f)
        
        # 保存菌株映射和embedding
        bacteria_mapping = {
            'bacteria_to_id': bacteria_to_id,
            'id_to_bacteria': id_to_bacteria,
            'bacteria_embedding_matrix': bacteria_embedding_matrix,
            'embedding_dim': bacteria_embedding_matrix.shape[1]
        }
        
        with open(os.path.join(self.features_output_dir, 'bacteria_mapping.pkl'), 'wb') as f:
            pickle.dump(bacteria_mapping, f)
        
        print(f"\n特征工程完成！所有文件保存在: {self.features_output_dir}")
        
        # 8. 生成特征工程报告
        self.generate_feature_report(datasets, seq_features_mapping, bacteria_mapping)
    
    def generate_feature_report(self, datasets, seq_features_mapping, bacteria_mapping):
        """
        生成特征工程报告
        """
        report_file = os.path.join(self.features_output_dir, 'feature_engineering_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 特征工程报告\n\n")
            
            f.write("## 数据集统计\n")
            for dataset_name, df in datasets.items():
                f.write(f"- {dataset_name}: {len(df)} 样本\n")
            f.write("\n")
            
            f.write("## 特征维度\n")
            f.write(f"- PLM embeddings: {seq_features_mapping['plm_embedding_dim']} 维\n")
            f.write(f"- 理化特征: {seq_features_mapping['physchem_feature_dim']} 维\n")
            f.write(f"- 菌株embedding: {bacteria_mapping['embedding_dim']} 维\n")
            f.write(f"- 总序列数: {len(seq_features_mapping['seq_to_plm'])}\n")
            f.write(f"- 总菌株数: {len(bacteria_mapping['bacteria_to_id'])}\n\n")
            
            f.write("## 理化特征列表\n")
            feature_names = [
                "序列长度", "分子量", "等电点", "GRAVY疏水性", "净电荷", "RK比例", "疏水矩",
                "正电荷比例", "负电荷比例", "极性残基比例"
            ] + [f"氨基酸_{aa}_比例" for aa in 'ACDEFGHIKLMNPQRSTVWY']
            
            for i, name in enumerate(feature_names):
                f.write(f"{i+1}. {name}\n")
            f.write("\n")
            
            f.write("## Top 10 菌株\n")
            bacteria_counts = pd.Series(list(bacteria_mapping['bacteria_to_id'].keys())).value_counts()
            for i, (bacteria, _) in enumerate(bacteria_counts.head(10).items(), 1):
                f.write(f"{i}. {bacteria}\n")
            
        print(f"特征工程报告已保存: {report_file}")

def main():
    """主函数"""
    # 配置参数
    processed_data_dir = "/Users/ricardozhao/PycharmProjects/AMP/processed_data"
    features_output_dir = "/Users/ricardozhao/PycharmProjects/AMP/features"
    
    # 创建特征提取器
    extractor = AMP_FeatureExtractor(
        processed_data_dir=processed_data_dir,
        features_output_dir=features_output_dir,
        esm_model_name='facebook/esm2_t33_650M_UR50D',  # ESM-2 650M参数版本
        device='auto'
    )
    
    # 执行特征工程
    extractor.process_all_datasets()

if __name__ == "__main__":
    main()
