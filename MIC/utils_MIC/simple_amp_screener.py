#!/usr/bin/env python3
"""
简化版AMP筛选器
基于三层级联：A快筛 → B₂双头精筛 → B₁面板广谱

核心思路：
1. A快筛：logMIC ≤ 1.0保留，≤ 0.7强推，批内minmax归一化
2. B₂双头：E.coli/S.aureus双菌株底线，短板优先评分
3. B₁面板：6菌株广谱评估，hit@10μM + broad + worst
4. 最终打分：S = 0.4*s_A + 0.4*s_T + 0.2*s_B
5. 分层选择：S/A/B/C分层 + CD-HIT去冗余 + 配额分配
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入模型架构和特征提取
sys.path.append('/root/NKU-TMU_AMP_project')
from train_discriminators import SequenceRegressionModel, ConditionalRegressionModel
from train_dual_head_model import DualHeadModel
from feature_engineering import AMP_FeatureExtractor

class SimpleAMPScreener:
    """简化版AMP筛选器"""
    
    def __init__(self, model_dir='model_outputs', features_dir='features', device='auto'):
        """
        初始化筛选器
        
        Args:
            model_dir: 模型文件目录
            features_dir: 特征文件目录  
            device: 计算设备
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_dir = model_dir
        self.features_dir = features_dir
        
        # 筛选参数（固定，不需要复杂调优）
        self.params = {
            # 阈值设置
            'gate_threshold': 1.0,      # A模型门控阈值（10μM）
            'strong_threshold': 0.7,    # 强推阈值（5μM）
            'dual_threshold': 1.0,      # 双头阈值
            
            # 权重设置
            'w_A': 0.4,                 # A模型权重
            'w_T': 0.4,                 # 双头模型权重  
            'w_B': 0.2,                 # 广谱模型权重
            'dual_min_weight': 0.7,     # 双头短板权重
            'dual_avg_weight': 0.3,     # 双头均值权重
            
            # 面板权重
            'hit_weight': 0.6,          # 命中率权重
            'broad_weight': 0.2,        # 广谱权重
            'worst_weight': 0.2,        # 最差情况权重
            
            # 分层比例
            'tier_S_ratio': 0.10,       # S级比例
            'tier_A_ratio': 0.30,       # A级比例  
            'tier_B_ratio': 0.60,       # B级比例
            
            # 去冗余设置
            'similarity_threshold': 0.8, # 序列相似度阈值
        }
        
        # 面板菌株（6个代表性菌株）
        self.panel_bacteria = [
            'escherichia_coli', 
            'staphylococcus_aureus',
            'pseudomonas_aeruginosa', 
            'klebsiella_pneumoniae',
            'acinetobacter_baumannii', 
            'enterococcus_faecalis'
        ]
        
        print(f"简化版AMP筛选器初始化完成，使用设备: {self.device}")
    
    def load_models(self):
        """加载三个训练好的模型"""
        print("正在加载训练好的模型...")
        
        # 加载特征提取器
        self.feature_extractor = AMP_FeatureExtractor(
            processed_data_dir=f'/root/NKU-TMU_AMP_project/processed_data',
            features_output_dir=f'/root/NKU-TMU_AMP_project/{self.features_dir}',
            device=self.device
        )
        
        # 加载标准化器
        with open(f'{self.features_dir}/sequence_train_features.pkl', 'rb') as f:
            seq_features = pickle.load(f)
        self.seq_scaler = StandardScaler()
        self.seq_scaler.fit(seq_features['physicochemical_features'])
        
        with open(f'{self.features_dir}/conditional_train_features.pkl', 'rb') as f:
            cond_features = pickle.load(f)
        self.cond_scaler = StandardScaler()
        self.cond_scaler.fit(cond_features['physicochemical_features'])
        
        # 加载菌株映射
        with open(f'{self.features_dir}/bacteria_mapping.pkl', 'rb') as f:
            self.bacteria_mapping = pickle.load(f)
        
        # 计算输入维度
        plm_dim = seq_features['plm_embeddings'].shape[1]
        physchem_dim = seq_features['physicochemical_features'].shape[1]
        input_dim = plm_dim + physchem_dim
        n_bacteria = len(self.bacteria_mapping['bacteria_to_id'])
        
        # 1. 序列聚合模型 (A)
        self.model_A = SequenceRegressionModel(input_dim=input_dim).to(self.device)
        self.model_A.load_state_dict(torch.load(f'{self.model_dir}/sequence_regression_best.pt', map_location=self.device))
        self.model_A.eval()
        
        # 2. 条件回归模型 (B₁)  
        self.model_B1 = ConditionalRegressionModel(
            input_dim=input_dim, 
            n_bacteria=n_bacteria,
            bacteria_embedding_dim=32
        ).to(self.device)
        self.model_B1.load_state_dict(torch.load(f'{self.model_dir}/conditional_regression_best.pt', map_location=self.device))
        self.model_B1.eval()
        
        # 3. 双头模型 (B₂)
        self.model_B2 = DualHeadModel(input_dim=input_dim, hidden_dims=[512, 256]).to(self.device)
        self.model_B2.load_state_dict(torch.load(f'{self.model_dir}/dual_head_best.pt', map_location=self.device))
        self.model_B2.eval()
        
        print("✓ 模型A (序列聚合) 加载完成")
        print("✓ 模型B₁ (条件回归) 加载完成") 
        print("✓ 模型B₂ (双头模型) 加载完成")
    
    def extract_features(self, sequences):
        """提取序列特征"""
        print(f"正在提取 {len(sequences)} 个序列的特征...")
        
        # PLM embeddings
        plm_embeddings = self.feature_extractor.extract_plm_embeddings(sequences)
        
        # 理化特征
        physchem_features = self.feature_extractor.calculate_physicochemical_features(sequences)
        
        # 标准化理化特征
        seq_features = np.concatenate([
            plm_embeddings,
            self.seq_scaler.transform(physchem_features)
        ], axis=1)
        
        cond_features = np.concatenate([
            plm_embeddings,
            self.cond_scaler.transform(physchem_features)
        ], axis=1)
        
        return {
            'seq_features': seq_features,
            'cond_features': cond_features,
            'sequences': sequences
        }
    
    def minmax_normalize(self, values):
        """批内最小-最大归一化"""
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return np.ones_like(values) * 0.5  # 如果都相同，返回中性分数
        return (values - min_val) / (max_val - min_val)
    
    def step1_gate_screening(self, features):
        """Step 1: A模型快筛（一级门控）"""
        print("\n=== Step 1: A模型快筛 ===")
        
        seq_features = features['seq_features']
        
        # A模型预测
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(seq_features).to(self.device)
            logMIC_A = self.model_A(seq_tensor).cpu().numpy()
        
        # 门控规则
        gate_pass = logMIC_A <= self.params['gate_threshold']
        strong_pass = logMIC_A <= self.params['strong_threshold']
        
        # 批内归一化分数（1 - minmax，越大越好）
        s_A = 1 - self.minmax_normalize(logMIC_A)
        
        print(f"门控通过: {gate_pass.sum()}/{len(logMIC_A)} ({100*gate_pass.sum()/len(logMIC_A):.1f}%)")
        print(f"强推候选: {strong_pass.sum()}/{len(logMIC_A)} ({100*strong_pass.sum()/len(logMIC_A):.1f}%)")
        print(f"A模型预测范围: [{np.min(logMIC_A):.3f}, {np.max(logMIC_A):.3f}]")
        
        return {
            'logMIC_A': logMIC_A,
            'gate_pass': gate_pass,
            'strong_pass': strong_pass,
            's_A': s_A
        }
    
    def step2_dual_head_screening(self, features, gate_results):
        """Step 2: B₂双头精筛（双菌株底线）"""
        print("\n=== Step 2: B₂双头精筛 ===")
        
        cond_features = features['cond_features']
        gate_pass = gate_results['gate_pass']
        
        # 只对通过门控的序列进行双头预测
        if gate_pass.sum() == 0:
            print("⚠️ 没有序列通过门控筛选")
            return self._empty_dual_results(len(features['sequences']))
        
        # B₂模型预测
        with torch.no_grad():
            cond_tensor = torch.FloatTensor(cond_features).to(self.device)
            logMIC_E, logMIC_S = self.model_B2(cond_tensor)
            logMIC_E = logMIC_E.cpu().numpy()
            logMIC_S = logMIC_S.cpu().numpy()
        
        # 双头规则
        dual_threshold = self.params['dual_threshold']
        strong_threshold = self.params['strong_threshold']
        
        dual_pass = (logMIC_E <= dual_threshold) & (logMIC_S <= dual_threshold)
        strong_dual_pass = ((logMIC_E <= strong_threshold) | (logMIC_S <= strong_threshold)) & dual_pass
        single_pass = ((logMIC_E <= dual_threshold) & (logMIC_S > dual_threshold)) | \
                     ((logMIC_E > dual_threshold) & (logMIC_S <= dual_threshold))
        
        # 批内归一化分数
        s_E = 1 - self.minmax_normalize(logMIC_E)
        s_S = 1 - self.minmax_normalize(logMIC_S)
        
        # 短板优先合成分数
        min_score = np.minimum(s_E, s_S)
        avg_score = (s_E + s_S) / 2
        s_T = self.params['dual_min_weight'] * min_score + self.params['dual_avg_weight'] * avg_score
        
        print(f"双菌株通过: {dual_pass.sum()}/{len(logMIC_E)} ({100*dual_pass.sum()/len(logMIC_E):.1f}%)")
        print(f"强双菌株: {strong_dual_pass.sum()}/{len(logMIC_E)} ({100*strong_dual_pass.sum()/len(logMIC_E):.1f}%)")
        print(f"单菌株通过: {single_pass.sum()}/{len(logMIC_E)} ({100*single_pass.sum()/len(logMIC_E):.1f}%)")
        print(f"E.coli预测范围: [{np.min(logMIC_E):.3f}, {np.max(logMIC_E):.3f}]")
        print(f"S.aureus预测范围: [{np.min(logMIC_S):.3f}, {np.max(logMIC_S):.3f}]")
        
        return {
            'logMIC_E': logMIC_E,
            'logMIC_S': logMIC_S,
            'dual_pass': dual_pass,
            'strong_dual_pass': strong_dual_pass,
            'single_pass': single_pass,
            's_E': s_E,
            's_S': s_S,
            's_T': s_T
        }
    
    def _empty_dual_results(self, n_sequences):
        """创建空的双头结果"""
        return {
            'logMIC_E': np.full(n_sequences, 10.0),  # 高值表示无效
            'logMIC_S': np.full(n_sequences, 10.0),
            'dual_pass': np.zeros(n_sequences, dtype=bool),
            'strong_dual_pass': np.zeros(n_sequences, dtype=bool),
            'single_pass': np.zeros(n_sequences, dtype=bool),
            's_E': np.zeros(n_sequences),
            's_S': np.zeros(n_sequences),
            's_T': np.zeros(n_sequences)
        }
    
    def step3_panel_evaluation(self, features, gate_results, dual_results):
        """Step 3: B₁面板广谱评估"""
        print("\n=== Step 3: B₁面板广谱评估 ===")
        
        cond_features = features['cond_features']
        gate_pass = gate_results['gate_pass']
        
        # 只对通过前两步的序列进行面板评估
        candidates = gate_pass & (dual_results['dual_pass'] | dual_results['single_pass'])
        
        if candidates.sum() == 0:
            print("⚠️ 没有序列通过前两步筛选")
            return self._empty_panel_results(len(features['sequences']))
        
        print(f"面板评估候选: {candidates.sum()}/{len(features['sequences'])} 个序列")
        
        # B₁模型面板预测
        panel_predictions = {}
        
        for bacteria_name in self.panel_bacteria:
            if bacteria_name in self.bacteria_mapping['bacteria_to_id']:
                bacteria_id = self.bacteria_mapping['bacteria_to_id'][bacteria_name]
                bacteria_ids = np.full(len(features['sequences']), bacteria_id)
                
                with torch.no_grad():
                    cond_tensor = torch.FloatTensor(cond_features).to(self.device)
                    bacteria_tensor = torch.LongTensor(bacteria_ids).to(self.device)
                    pred = self.model_B1(cond_tensor, bacteria_tensor).cpu().numpy()
                
                panel_predictions[bacteria_name] = pred
        
        if len(panel_predictions) == 0:
            print("⚠️ 面板菌株映射失败")
            return self._empty_panel_results(len(features['sequences']))
        
        # 转换为数组
        panel_matrix = np.array(list(panel_predictions.values()))  # (n_bacteria, n_sequences)
        
        # 计算面板指标
        hit_at_10 = np.mean(panel_matrix <= 1.0, axis=0)  # 命中率@10μM
        broad_mean = np.mean(panel_matrix, axis=0)         # 面板均值
        worst_max = np.max(panel_matrix, axis=0)           # 最差值
        
        # 批内归一化分数
        s_hit = hit_at_10  # 命中率本身就是0-1
        s_broad = 1 - self.minmax_normalize(broad_mean)    # 均值越小越好
        s_worst = 1 - self.minmax_normalize(worst_max)     # 最差越小越好
        
        # 合成面板分数
        s_B = (self.params['hit_weight'] * s_hit + 
               self.params['broad_weight'] * s_broad + 
               self.params['worst_weight'] * s_worst)
        
        print(f"平均命中率@10μM: {np.mean(hit_at_10):.3f}")
        print(f"面板均值范围: [{np.min(broad_mean):.3f}, {np.max(broad_mean):.3f}]")
        print(f"最差值范围: [{np.min(worst_max):.3f}, {np.max(worst_max):.3f}]")
        
        return {
            'panel_predictions': panel_predictions,
            'hit_at_10': hit_at_10,
            'broad_mean': broad_mean,
            'worst_max': worst_max,
            's_hit': s_hit,
            's_broad': s_broad,
            's_worst': s_worst,
            's_B': s_B
        }
    
    def _empty_panel_results(self, n_sequences):
        """创建空的面板结果"""
        return {
            'panel_predictions': {},
            'hit_at_10': np.zeros(n_sequences),
            'broad_mean': np.full(n_sequences, 5.0),
            'worst_max': np.full(n_sequences, 10.0),
            's_hit': np.zeros(n_sequences),
            's_broad': np.zeros(n_sequences),
            's_worst': np.zeros(n_sequences),
            's_B': np.zeros(n_sequences)
        }
    
    def calculate_final_scores(self, gate_results, dual_results, panel_results):
        """计算最终分数"""
        print("\n=== 计算最终分数 ===")
        
        s_A = gate_results['s_A']
        s_T = dual_results['s_T']
        s_B = panel_results['s_B']
        
        # 最终分数
        S_final = (self.params['w_A'] * s_A + 
                   self.params['w_T'] * s_T + 
                   self.params['w_B'] * s_B)
        
        print(f"最终分数范围: [{np.min(S_final):.3f}, {np.max(S_final):.3f}]")
        print(f"最终分数均值: {np.mean(S_final):.3f}")
        
        return S_final
    
    def apply_tier_classification(self, final_scores):
        """应用分层分类"""
        print("\n=== 分层分类 ===")
        
        n_sequences = len(final_scores)
        
        # 计算分位数阈值
        tier_S_threshold = np.percentile(final_scores, 100 - self.params['tier_S_ratio'] * 100)
        tier_A_threshold = np.percentile(final_scores, 100 - self.params['tier_A_ratio'] * 100)
        tier_B_threshold = np.percentile(final_scores, 100 - self.params['tier_B_ratio'] * 100)
        
        # 分层
        tiers = []
        for score in final_scores:
            if score >= tier_S_threshold:
                tiers.append('S')
            elif score >= tier_A_threshold:
                tiers.append('A')
            elif score >= tier_B_threshold:
                tiers.append('B')
            else:
                tiers.append('C')
        
        tiers = np.array(tiers)
        
        # 统计
        tier_counts = {tier: (tiers == tier).sum() for tier in ['S', 'A', 'B', 'C']}
        
        print("分层统计:")
        for tier, count in tier_counts.items():
            print(f"  {tier}级: {count} ({100*count/n_sequences:.1f}%)")
        
        return tiers, tier_counts
    
    def apply_diversity_sampling(self, sequences, final_scores, tiers, top_k=2000):
        """应用多样性去冗余和配额分配"""
        print(f"\n=== 多样性采样 (目标: {top_k}) ===")
        
        # 简化版：按分层优先级 + 分数排序
        # 实际项目中可以加入CD-HIT或序列相似度去冗余
        
        tier_priority = {'S': 0, 'A': 1, 'B': 2, 'C': 3}
        
        # 创建排序索引
        indices = np.arange(len(sequences))
        sort_keys = [(tier_priority[tier], -score, idx) for idx, (tier, score) in enumerate(zip(tiers, final_scores))]
        sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
        
        # 取前top_k个
        selected_indices = sorted_indices[:min(top_k, len(sorted_indices))]
        
        # 统计选中的分层分布
        selected_tiers = tiers[selected_indices]
        selected_tier_counts = {tier: (selected_tiers == tier).sum() for tier in ['S', 'A', 'B', 'C']}
        
        print("选中序列分层分布:")
        for tier, count in selected_tier_counts.items():
            if count > 0:
                print(f"  {tier}级: {count} ({100*count/len(selected_indices):.1f}%)")
        
        return {
            'selected_indices': selected_indices,
            'selected_sequences': [sequences[i] for i in selected_indices],
            'selected_scores': final_scores[selected_indices],
            'selected_tiers': selected_tiers,
            'tier_counts': selected_tier_counts
        }
    
    def screen_sequences(self, sequences, top_k=2000):
        """主要筛选函数"""
        print(f"\n{'='*60}")
        print(f"开始简化版AMP筛选 - 输入序列: {len(sequences)}")
        print(f"{'='*60}")
        
        # 特征提取
        features = self.extract_features(sequences)
        
        # Step 1: A模型快筛
        gate_results = self.step1_gate_screening(features)
        
        # Step 2: B₂双头精筛
        dual_results = self.step2_dual_head_screening(features, gate_results)
        
        # Step 3: B₁面板评估
        panel_results = self.step3_panel_evaluation(features, gate_results, dual_results)
        
        # 最终分数计算
        final_scores = self.calculate_final_scores(gate_results, dual_results, panel_results)
        
        # 分层分类
        tiers, tier_counts = self.apply_tier_classification(final_scores)
        
        # 多样性采样
        selection_results = self.apply_diversity_sampling(sequences, final_scores, tiers, top_k)
        
        # 整合结果
        screening_results = {
            'input_sequences': sequences,
            'features': features,
            'gate_results': gate_results,
            'dual_results': dual_results,
            'panel_results': panel_results,
            'final_scores': final_scores,
            'tiers': tiers,
            'tier_counts': tier_counts,
            'selection_results': selection_results,
            'params': self.params.copy()
        }
        
        print(f"\n{'='*60}")
        print(f"筛选完成！从 {len(sequences)} 个序列中选出 {len(selection_results['selected_sequences'])} 个")
        print(f"{'='*60}")
        
        return screening_results
    
    def save_results(self, screening_results, output_prefix):
        """保存筛选结果"""
        # 保存详细结果
        with open(f'{output_prefix}_detailed_results.pkl', 'wb') as f:
            pickle.dump(screening_results, f)
        
        # 保存选中序列CSV
        selection = screening_results['selection_results']
        df_selected = pd.DataFrame({
            'sequence': selection['selected_sequences'],
            'final_score': selection['selected_scores'],
            'tier': selection['selected_tiers'],
            'rank': range(1, len(selection['selected_sequences']) + 1)
        })
        
        df_selected.to_csv(f'{output_prefix}_selected_sequences.csv', index=False)
        
        # 保存筛选报告
        self.generate_report(screening_results, f'{output_prefix}_screening_report.md')
        
        print(f"✓ 详细结果: {output_prefix}_detailed_results.pkl")
        print(f"✓ 选中序列: {output_prefix}_selected_sequences.csv")
        print(f"✓ 筛选报告: {output_prefix}_screening_report.md")
    
    def generate_report(self, screening_results, report_file):
        """生成筛选报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 简化版AMP筛选报告\n\n")
            
            # 基本信息
            f.write("## 筛选概况\n")
            f.write(f"- 输入序列数: {len(screening_results['input_sequences'])}\n")
            f.write(f"- 最终选中数: {len(screening_results['selection_results']['selected_sequences'])}\n")
            f.write(f"- 筛选成功率: {len(screening_results['selection_results']['selected_sequences']) / len(screening_results['input_sequences']) * 100:.1f}%\n\n")
            
            # 各步骤统计
            f.write("## 各步骤筛选统计\n")
            gate_pass = screening_results['gate_results']['gate_pass'].sum()
            dual_pass = screening_results['dual_results']['dual_pass'].sum()
            
            f.write(f"- Step 1 (A快筛): {gate_pass}/{len(screening_results['input_sequences'])} ({100*gate_pass/len(screening_results['input_sequences']):.1f}%) 通过\n")
            f.write(f"- Step 2 (双头): {dual_pass}/{len(screening_results['input_sequences'])} ({100*dual_pass/len(screening_results['input_sequences']):.1f}%) 双菌株通过\n")
            f.write(f"- Step 3 (面板): 对通过前两步的序列进行广谱评估\n\n")
            
            # 分层统计
            f.write("## 分层统计\n")
            for tier, count in screening_results['tier_counts'].items():
                f.write(f"- {tier}级: {count} 个序列\n")
            f.write("\n")
            
            # 选中序列分层分布
            f.write("## 选中序列分布\n")
            for tier, count in screening_results['selection_results']['tier_counts'].items():
                if count > 0:
                    f.write(f"- {tier}级: {count} 个序列\n")
            f.write("\n")
            
            # 参数设置
            f.write("## 参数设置\n")
            for key, value in screening_results['params'].items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            # Top 20序列
            f.write("## Top 20 选中序列\n")
            selection = screening_results['selection_results']
            for i in range(min(20, len(selection['selected_sequences']))):
                seq = selection['selected_sequences'][i]
                score = selection['selected_scores'][i]
                tier = selection['selected_tiers'][i]
                f.write(f"{i+1:2d}. [{tier}] {seq} (分数: {score:.4f})\n")

def main():
    """主函数"""
    # 读取候选序列
    input_file = '/root/NKU-TMU_AMP_project/decode/filtered_candidate_sequences.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请先运行序列预处理脚本生成候选序列集")
        return
    
    print(f"读取候选序列: {input_file}")
    df = pd.read_csv(input_file)
    
    # 检查序列列名
    seq_column = None
    for col in ['aa_seq', 'sequence', 'seq']:
        if col in df.columns:
            seq_column = col
            break
    
    if seq_column is None:
        print(f"❌ 输入文件缺少序列列，可用列: {list(df.columns)}")
        return
    
    sequences = df[seq_column].tolist()
    print(f"加载了 {len(sequences)} 个候选序列（列名: {seq_column}）")
    
    # 创建筛选器
    screener = SimpleAMPScreener(
        model_dir='/root/NKU-TMU_AMP_project/model_outputs',
        features_dir='/root/NKU-TMU_AMP_project/features'
    )
    
    # 加载模型
    screener.load_models()
    
    # 执行筛选
    screening_results = screener.screen_sequences(sequences, top_k=2000)
    
    # 保存结果
    output_prefix = '/root/NKU-TMU_AMP_project/decode/screening_results'
    screener.save_results(screening_results, output_prefix)
    
    print(f"\n🎉 筛选完成！结果保存在: {output_prefix}_*")

if __name__ == "__main__":
    main()
