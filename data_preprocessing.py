#!/usr/bin/env python3
"""
GRAMPA数据集预处理脚本
根据筛选器设计.md的要求进行数据清洗与标准化
"""

import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class GRAMPAPreprocessor:
    def __init__(self, input_file, output_dir='processed_data'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.processed_df = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 标准氨基酸字母表
        self.standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        # 菌株名称标准化映射
        self.bacteria_mapping = {
            'escherichia coli': 'escherichia_coli',
            'e. coli': 'escherichia_coli',
            'e.coli': 'escherichia_coli',
            'staphylococcus aureus': 'staphylococcus_aureus',
            'staph. aureus': 'staphylococcus_aureus',
            's. aureus': 'staphylococcus_aureus',
            's.aureus': 'staphylococcus_aureus',
            'pseudomonas aeruginosa': 'pseudomonas_aeruginosa',
            'p. aeruginosa': 'pseudomonas_aeruginosa',
            'p.aeruginosa': 'pseudomonas_aeruginosa',
            'klebsiella pneumoniae': 'klebsiella_pneumoniae',
            'k. pneumoniae': 'klebsiella_pneumoniae',
            'k.pneumoniae': 'klebsiella_pneumoniae',
            'candida albicans': 'candida_albicans',
            'c. albicans': 'candida_albicans',
            'c.albicans': 'candida_albicans',
            'bacillus subtilis': 'bacillus_subtilis',
            'b. subtilis': 'bacillus_subtilis',
            'enterococcus faecalis': 'enterococcus_faecalis',
            'e. faecalis': 'enterococcus_faecalis',
            'streptococcus pyogenes': 'streptococcus_pyogenes',
            's. pyogenes': 'streptococcus_pyogenes',
        }
    
    def load_data(self):
        """加载数据并进行基本分析"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.input_file)
        
        print(f"数据集形状: {self.df.shape}")
        print(f"列名: {self.df.columns.tolist()}")
        print(f"缺失值统计:")
        print(self.df.isnull().sum())
        print(f"\n数据类型:")
        print(self.df.dtypes)
        print(f"\nvalue字段统计:")
        print(self.df['value'].describe())
        print(f"\ncensor字段统计:")
        print(self.df['censor'].value_counts(dropna=False))
        
        return self.df
    
    def normalize_sequence(self, sequence):
        """序列合法化处理"""
        if pd.isna(sequence):
            return None, 1.0  # 返回序列和X占比
            
        # 转为大写
        seq = str(sequence).upper().strip()
        
        # 移除非字母字符（空格、标点等）
        seq = re.sub(r'[^A-Z]', '', seq)
        
        if len(seq) == 0:
            return None, 1.0
            
        # 将非标准氨基酸映射为X
        normalized_seq = ''
        for aa in seq:
            if aa in self.standard_aa:
                normalized_seq += aa
            else:
                normalized_seq += 'X'
        
        # 计算X占比
        x_ratio = normalized_seq.count('X') / len(normalized_seq) if len(normalized_seq) > 0 else 1.0
        
        return normalized_seq, x_ratio
    
    def normalize_bacterium(self, bacterium_name):
        """菌株名称标准化"""
        if pd.isna(bacterium_name):
            return 'unknown'
            
        name = str(bacterium_name).lower().strip()
        
        # 移除株系信息（括号内容、数字、特殊符号等）
        name = re.sub(r'\([^)]*\)', '', name)  # 移除括号内容
        name = re.sub(r'\s+\d+.*$', '', name)  # 移除数字及后续内容
        name = re.sub(r'\s+strain.*$', '', name, flags=re.IGNORECASE)  # 移除strain信息
        name = re.sub(r'\s+atcc.*$', '', name, flags=re.IGNORECASE)  # 移除ATCC信息
        name = name.strip()
        
        # 标准化映射
        if name in self.bacteria_mapping:
            return self.bacteria_mapping[name]
        
        # 对于未映射的，保留属种名
        parts = name.split()
        if len(parts) >= 2:
            genus_species = f"{parts[0]}_{parts[1]}"
            return genus_species.replace(' ', '_').replace('-', '_')
        
        return name.replace(' ', '_').replace('-', '_')
    
    def sequence_cleaning(self):
        """步骤1: 序列合法化"""
        print("\n=== 步骤1: 序列合法化 ===")
        
        # 处理序列
        seq_info = self.df['sequence'].apply(self.normalize_sequence)
        self.df['normalized_sequence'] = [x[0] for x in seq_info]
        self.df['x_ratio'] = [x[1] for x in seq_info]
        
        # 统计
        print(f"原始样本数: {len(self.df)}")
        
        # 移除空序列
        valid_seq_mask = self.df['normalized_sequence'].notna()
        print(f"空序列样本数: {(~valid_seq_mask).sum()}")
        
        # 移除X占比>10%的序列
        x_ratio_mask = self.df['x_ratio'] <= 0.1
        print(f"X占比>10%的样本数: {(~x_ratio_mask).sum()}")
        
        # 长度筛选
        self.df['seq_length'] = self.df['normalized_sequence'].fillna('').str.len()
        length_5_48_mask = (self.df['seq_length'] >= 5) & (self.df['seq_length'] <= 48)
        length_lt5_mask = (self.df['seq_length'] > 0) & (self.df['seq_length'] < 5)
        length_gt48_mask = self.df['seq_length'] > 48
        
        print(f"长度<5的样本数: {length_lt5_mask.sum()}")
        print(f"长度5-48的样本数: {length_5_48_mask.sum()}")
        print(f"长度>48的样本数: {length_gt48_mask.sum()}")
        
        # 主训练集：5-48 aa，X占比<=10%
        main_mask = valid_seq_mask & x_ratio_mask & length_5_48_mask
        self.df['dataset_split'] = 'exclude'
        self.df.loc[main_mask, 'dataset_split'] = 'main'
        self.df.loc[valid_seq_mask & x_ratio_mask & length_lt5_mask, 'dataset_split'] = 'short'
        self.df.loc[valid_seq_mask & x_ratio_mask & length_gt48_mask, 'dataset_split'] = 'long'
        
        print(f"主训练集样本数: {main_mask.sum()}")
        print(f"短肽样本数: {(self.df['dataset_split'] == 'short').sum()}")
        print(f"长肽样本数: {(self.df['dataset_split'] == 'long').sum()}")
        print(f"排除样本数: {(self.df['dataset_split'] == 'exclude').sum()}")
        
        return self.df
    
    def bacteria_normalization(self):
        """步骤2: 菌株归一化"""
        print("\n=== 步骤2: 菌株归一化 ===")
        
        # 标准化菌株名
        self.df['normalized_bacterium'] = self.df['bacterium'].apply(self.normalize_bacterium)
        
        # 统计菌株频次
        bacteria_counts = self.df['normalized_bacterium'].value_counts()
        print(f"唯一菌株数: {len(bacteria_counts)}")
        print(f"Top 10菌株:")
        print(bacteria_counts.head(10))
        
        # 长尾菌株处理（频次<10的归为other）
        low_freq_bacteria = bacteria_counts[bacteria_counts < 10].index
        print(f"低频菌株数（<10次）: {len(low_freq_bacteria)}")
        
        self.df['final_bacterium'] = self.df['normalized_bacterium'].copy()
        self.df.loc[self.df['normalized_bacterium'].isin(low_freq_bacteria), 'final_bacterium'] = 'other'
        
        final_bacteria_counts = self.df['final_bacterium'].value_counts()
        print(f"最终菌株数: {len(final_bacteria_counts)}")
        print(f"归为other的样本数: {(self.df['final_bacterium'] == 'other').sum()}")
        
        return self.df
    
    def handle_duplicates(self):
        """步骤3: 重复测定处理"""
        print("\n=== 步骤3: 重复测定处理 ===")
        
        # 只处理主训练集
        main_df = self.df[self.df['dataset_split'] == 'main'].copy()
        
        # 统计重复情况
        duplicate_groups = main_df.groupby(['normalized_sequence', 'final_bacterium'])
        duplicate_stats = duplicate_groups.size()
        
        print(f"唯一(sequence, bacterium)对数: {len(duplicate_stats)}")
        print(f"重复测定统计:")
        print(duplicate_stats.value_counts().sort_index())
        print(f"最大重复次数: {duplicate_stats.max()}")
        
        # 对每个(sequence, bacterium)组合计算几何均值
        def geometric_mean_log(values):
            """对log值计算几何均值（先转回原值，算几何均值，再取log）"""
            # value是log10(uM)，需要转回uM
            original_values = 10 ** values
            # 计算几何均值
            geom_mean = np.exp(np.mean(np.log(original_values)))
            # 转回log10
            return np.log10(geom_mean)
        
        aggregated_data = []
        for (seq, bact), group in duplicate_groups:
            values = group['value'].values
            n_measurements = len(values)
            
            # 删失信息处理 - 改进版
            censor_info = group['censor'].fillna('').values
            censored_mask = censor_info == '>'
            has_censoring = censored_mask.any()
            
            # 计算聚合值
            if has_censoring:
                # 有删失的情况：分别处理删失和非删失值
                censored_values = values[censored_mask]
                uncensored_values = values[~censored_mask]
                
                # 删失阈值：删失样本中的最大值作为下界约束
                censoring_threshold = censored_values.max() if len(censored_values) > 0 else None
                
                # 聚合值：只用非删失值计算几何均值
                if len(uncensored_values) > 0:
                    agg_value = geometric_mean_log(uncensored_values)
                else:
                    # 全是删失值的情况，用删失阈值作为下界估计
                    agg_value = censoring_threshold
                    
                # 标准差：只用非删失值
                value_std = np.std(uncensored_values) if len(uncensored_values) > 1 else 0.0
                
            else:
                # 无删失的情况：正常处理
                agg_value = geometric_mean_log(values)
                value_std = np.std(values)
                censoring_threshold = None
            
            # 其他信息
            database = group['database'].iloc[0]
            unit = group['unit'].iloc[0]
            seq_length = group['seq_length'].iloc[0]
            x_ratio = group['x_ratio'].iloc[0]
            
            aggregated_data.append({
                'sequence': seq,
                'bacterium': bact,
                'value': agg_value,
                'n_measurements': n_measurements,
                'n_censored': censored_mask.sum(),
                'n_uncensored': (~censored_mask).sum(),
                'value_std': value_std,
                'has_censoring': has_censoring,
                'censoring_threshold': censoring_threshold,  # 删失下界约束
                'unit': unit,
                'database': database,
                'seq_length': seq_length,
                'x_ratio': x_ratio
            })
        
        self.aggregated_df = pd.DataFrame(aggregated_data)
        print(f"聚合后样本数: {len(self.aggregated_df)}")
        print(f"平均重复测定次数: {self.aggregated_df['n_measurements'].mean():.2f}")
        print(f"删失样本数: {self.aggregated_df['has_censoring'].sum()}")
        print(f"完全删失样本数: {(self.aggregated_df['n_uncensored'] == 0).sum()}")
        
        return self.aggregated_df
    
    def handle_censoring(self):
        """步骤4: 删失样本处理 - 改进版"""
        print("\n=== 步骤4: 删失样本处理（改进版） ===")
        
        censored_count = self.aggregated_df['has_censoring'].sum()
        fully_censored_count = (self.aggregated_df['n_uncensored'] == 0).sum()
        
        print(f"包含删失信息的样本数: {censored_count}")
        print(f"完全删失样本数: {fully_censored_count}")
        print(f"删失样本占比: {censored_count / len(self.aggregated_df) * 100:.2f}%")
        
        # 删失信息统计
        if censored_count > 0:
            censored_df = self.aggregated_df[self.aggregated_df['has_censoring']]
            print(f"删失阈值分布:")
            print(f"  - 最小删失阈值: {censored_df['censoring_threshold'].min():.3f}")
            print(f"  - 最大删失阈值: {censored_df['censoring_threshold'].max():.3f}")
            print(f"  - 平均删失阈值: {censored_df['censoring_threshold'].mean():.3f}")
            
            # 检查删失一致性：聚合值不应低于删失阈值
            inconsistent_mask = censored_df['value'] < censored_df['censoring_threshold']
            inconsistent_count = inconsistent_mask.sum()
            if inconsistent_count > 0:
                print(f"警告: {inconsistent_count}个样本的聚合值低于删失阈值（数据不一致）")
        
        # 为损失函数准备删失标记
        self.aggregated_df['is_censored'] = self.aggregated_df['has_censoring']
        
        return self.aggregated_df
    
    def winsorize_values(self, percentile_range=(1, 99)):
        """步骤5: 异常值稳健化"""
        print("\n=== 步骤5: 异常值稳健化 ===")
        
        values = self.aggregated_df['value']
        
        # 计算分位数
        p_low = np.percentile(values, percentile_range[0])
        p_high = np.percentile(values, percentile_range[1])
        
        print(f"原始value范围: [{values.min():.3f}, {values.max():.3f}]")
        print(f"Winsorize范围 ({percentile_range[0]}%-{percentile_range[1]}%): [{p_low:.3f}, {p_high:.3f}]")
        
        # Winsorize处理
        winsorized_values = np.clip(values, p_low, p_high)
        
        # 统计影响的样本数
        affected_low = (values < p_low).sum()
        affected_high = (values > p_high).sum()
        print(f"被调整的样本数: 低端{affected_low}个, 高端{affected_high}个")
        
        self.aggregated_df['value_winsorized'] = winsorized_values
        self.aggregated_df['value_original'] = values
        
        return self.aggregated_df
    
    def create_sequence_aggregated_dataset(self):
        """创建序列聚合数据集（用于模型A）"""
        print("\n=== 创建序列聚合数据集 ===")
        
        # 按序列聚合，计算所有菌株的平均活性
        seq_groups = self.aggregated_df.groupby('sequence')
        
        seq_aggregated_data = []
        for seq, group in seq_groups:
            # 使用winsorized值计算均值
            mean_value = group['value_winsorized'].mean()
            std_value = group['value_winsorized'].std()
            n_bacteria = len(group)
            total_measurements = group['n_measurements'].sum()
            
            # 序列特征
            seq_length = group['seq_length'].iloc[0]
            x_ratio = group['x_ratio'].iloc[0]
            
            # 删失信息
            has_any_censoring = group['has_censoring'].any()
            
            seq_aggregated_data.append({
                'sequence': seq,
                'mean_log_mic': mean_value,
                'std_log_mic': std_value,
                'n_bacteria_tested': n_bacteria,
                'total_measurements': total_measurements,
                'seq_length': seq_length,
                'x_ratio': x_ratio,
                'has_censoring': has_any_censoring
            })
        
        self.seq_aggregated_df = pd.DataFrame(seq_aggregated_data)
        print(f"序列聚合数据集样本数: {len(self.seq_aggregated_df)}")
        
        return self.seq_aggregated_df
    
    def create_stratified_bacteria_splits(self):
        """创建按菌株分层的序列分组"""
        # 为每个序列计算主要菌株（出现最多的菌株）
        seq_bacteria_mapping = {}
        for seq in self.aggregated_df['sequence'].unique():
            seq_data = self.aggregated_df[self.aggregated_df['sequence'] == seq]
            main_bacterium = seq_data['bacterium'].value_counts().index[0]
            seq_bacteria_mapping[seq] = main_bacterium
        
        return seq_bacteria_mapping
    
    def create_train_val_test_splits(self, n_splits=5, test_size=0.2, val_size=0.1):
        """步骤6: 创建数据划分 - 改进版使用真正的GroupKFold"""
        print("\n=== 步骤6: 数据划分（改进版GroupKFold） ===")
        
        # 准备序列级数据用于划分
        sequences = self.aggregated_df['sequence'].unique()
        print(f"唯一序列数: {len(sequences)}")
        
        # 创建序列-主要菌株映射，用于分层
        seq_bacteria_mapping = self.create_stratified_bacteria_splits()
        
        # 统计每个菌株的序列数
        bacteria_seq_counts = pd.Series(seq_bacteria_mapping.values()).value_counts()
        print(f"各菌株的序列数分布 (Top 10):")
        print(bacteria_seq_counts.head(10))
        
        # 为了保证菌株分布平衡，我们使用分层策略
        # 首先按菌株频次分组
        high_freq_bacteria = bacteria_seq_counts[bacteria_seq_counts >= 50].index  # 高频菌株
        medium_freq_bacteria = bacteria_seq_counts[(bacteria_seq_counts >= 10) & (bacteria_seq_counts < 50)].index
        low_freq_bacteria = bacteria_seq_counts[bacteria_seq_counts < 10].index
        
        print(f"高频菌株数 (>=50序列): {len(high_freq_bacteria)}")
        print(f"中频菌株数 (10-49序列): {len(medium_freq_bacteria)}")  
        print(f"低频菌株数 (<10序列): {len(low_freq_bacteria)}")
        
        # 分别对每个频次组使用GroupKFold
        def stratified_group_split(sequences, bacteria_mapping, test_size, val_size, random_state=42):
            """按菌株分层的GroupKFold划分"""
            np.random.seed(random_state)
            
            # 按菌株分组序列
            bacteria_sequences = {}
            for seq, bacterium in bacteria_mapping.items():
                if bacterium not in bacteria_sequences:
                    bacteria_sequences[bacterium] = []
                bacteria_sequences[bacterium].append(seq)
            
            train_seqs, val_seqs, test_seqs = [], [], []
            
            # 对每个菌株的序列进行划分
            for bacterium, seqs in bacteria_sequences.items():
                seqs = np.array(seqs)
                n_seqs = len(seqs)
                
                if n_seqs == 1:
                    # 只有1个序列，随机分配
                    split_choice = np.random.choice(['train', 'val', 'test'], p=[1-test_size-val_size, val_size, test_size])
                    if split_choice == 'train':
                        train_seqs.extend(seqs)
                    elif split_choice == 'val':
                        val_seqs.extend(seqs)
                    else:
                        test_seqs.extend(seqs)
                elif n_seqs == 2:
                    # 2个序列，一个给train，一个随机分配给val或test
                    train_seqs.append(seqs[0])
                    split_choice = np.random.choice(['val', 'test'])
                    if split_choice == 'val':
                        val_seqs.append(seqs[1])
                    else:
                        test_seqs.append(seqs[1])
                else:
                    # 多个序列，按比例划分
                    shuffled = np.random.permutation(seqs)
                    n_test = max(1, int(n_seqs * test_size))
                    n_val = max(1, int(n_seqs * val_size))
                    n_train = n_seqs - n_test - n_val
                    
                    if n_train < 1:  # 确保至少有1个训练样本
                        n_train = 1
                        n_test = max(1, n_seqs - n_train - n_val)
                        n_val = n_seqs - n_train - n_test
                    
                    train_seqs.extend(shuffled[:n_train])
                    val_seqs.extend(shuffled[n_train:n_train+n_val])
                    test_seqs.extend(shuffled[n_train+n_val:])
            
            return set(train_seqs), set(val_seqs), set(test_seqs)
        
        # 执行分层划分
        train_sequences, val_sequences, test_sequences = stratified_group_split(
            sequences, seq_bacteria_mapping, test_size, val_size
        )
        
        print(f"训练集序列数: {len(train_sequences)}")
        print(f"验证集序列数: {len(val_sequences)}")
        print(f"测试集序列数: {len(test_sequences)}")
        
        # 验证没有重叠
        assert len(train_sequences & val_sequences) == 0, "训练集和验证集有重叠序列"
        assert len(train_sequences & test_sequences) == 0, "训练集和测试集有重叠序列"
        assert len(val_sequences & test_sequences) == 0, "验证集和测试集有重叠序列"
        
        # 为聚合数据集添加split标记
        def assign_split(seq):
            if seq in train_sequences:
                return 'train'
            elif seq in val_sequences:
                return 'val'
            else:
                return 'test'
        
        self.aggregated_df['split'] = self.aggregated_df['sequence'].apply(assign_split)
        self.seq_aggregated_df['split'] = self.seq_aggregated_df['sequence'].apply(assign_split)
        
        # 统计每个split的样本数和菌株分布
        split_stats = self.aggregated_df.groupby('split').agg({
            'sequence': 'count',
            'bacterium': 'nunique',
            'value_winsorized': ['mean', 'std']
        }).round(3)
        
        print(f"各split统计:")
        print(split_stats)
        
        # 检查菌株分布平衡性
        print(f"\n各split菌株分布平衡性检查:")
        for split in ['train', 'val', 'test']:
            split_bacteria = self.aggregated_df[self.aggregated_df['split'] == split]['bacterium'].value_counts()
            print(f"{split}集 Top 5菌株: {dict(split_bacteria.head(5))}")
        
        return train_sequences, val_sequences, test_sequences
    
    def save_processed_datasets(self):
        """保存处理后的数据集"""
        print("\n=== 保存数据集 ===")
        
        # 保存完整的聚合数据集
        output_file = os.path.join(self.output_dir, 'grampa_aggregated_full.csv')
        self.aggregated_df.to_csv(output_file, index=False)
        print(f"完整聚合数据集已保存: {output_file}")
        
        # 保存序列聚合数据集
        seq_output_file = os.path.join(self.output_dir, 'grampa_sequence_aggregated.csv')
        self.seq_aggregated_df.to_csv(seq_output_file, index=False)
        print(f"序列聚合数据集已保存: {seq_output_file}")
        
        # 按split保存
        for split in ['train', 'val', 'test']:
            # 条件回归数据集（包含菌株信息）
            split_df = self.aggregated_df[self.aggregated_df['split'] == split].copy()
            split_file = os.path.join(self.output_dir, f'grampa_conditional_{split}.csv')
            split_df.to_csv(split_file, index=False)
            print(f"{split}集（条件回归）已保存: {split_file} ({len(split_df)} 样本)")
            
            # 序列回归数据集
            seq_split_df = self.seq_aggregated_df[self.seq_aggregated_df['split'] == split].copy()
            seq_split_file = os.path.join(self.output_dir, f'grampa_sequence_{split}.csv')
            seq_split_df.to_csv(seq_split_file, index=False)
            print(f"{split}集（序列回归）已保存: {seq_split_file} ({len(seq_split_df)} 样本)")
        
        # 保存其他长度的数据集
        other_splits = self.df[self.df['dataset_split'].isin(['short', 'long'])].copy()
        if len(other_splits) > 0:
            other_file = os.path.join(self.output_dir, 'grampa_other_lengths.csv')
            other_splits.to_csv(other_file, index=False)
            print(f"其他长度数据集已保存: {other_file} ({len(other_splits)} 样本)")
        
        # 保存处理报告
        self.save_processing_report()
    
    def save_processing_report(self):
        """保存处理报告"""
        report_file = os.path.join(self.output_dir, 'processing_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GRAMPA数据集处理报告\n\n")
            
            f.write("## 原始数据统计\n")
            f.write(f"- 原始样本数: {len(self.df)}\n")
            f.write(f"- 唯一序列数: {self.df['normalized_sequence'].nunique()}\n")
            f.write(f"- 唯一菌株数: {self.df['normalized_bacterium'].nunique()}\n\n")
            
            f.write("## 序列长度分布\n")
            length_dist = self.df['dataset_split'].value_counts()
            for category, count in length_dist.items():
                f.write(f"- {category}: {count}\n")
            f.write("\n")
            
            f.write("## 聚合后数据统计\n")
            f.write(f"- 聚合样本数: {len(self.aggregated_df)}\n")
            f.write(f"- 序列聚合样本数: {len(self.seq_aggregated_df)}\n")
            f.write(f"- 平均重复测定次数: {self.aggregated_df['n_measurements'].mean():.2f}\n")
            f.write(f"- 删失样本数: {self.aggregated_df['has_censoring'].sum()}\n")
            f.write(f"- 完全删失样本数: {(self.aggregated_df['n_uncensored'] == 0).sum()}\n\n")
            
            f.write("## 数据划分统计\n")
            split_stats = self.aggregated_df['split'].value_counts()
            for split, count in split_stats.items():
                f.write(f"- {split}: {count}\n")
            f.write("\n")
            
            f.write("## Value分布统计\n")
            f.write(f"- 原始范围: [{self.aggregated_df['value_original'].min():.3f}, {self.aggregated_df['value_original'].max():.3f}]\n")
            f.write(f"- Winsorized范围: [{self.aggregated_df['value_winsorized'].min():.3f}, {self.aggregated_df['value_winsorized'].max():.3f}]\n")
            f.write(f"- 均值: {self.aggregated_df['value_winsorized'].mean():.3f}\n")
            f.write(f"- 标准差: {self.aggregated_df['value_winsorized'].std():.3f}\n\n")
            
            f.write("## 菌株分布（Top 10）\n")
            bacteria_counts = self.aggregated_df['bacterium'].value_counts().head(10)
            for bacterium, count in bacteria_counts.items():
                f.write(f"- {bacterium}: {count}\n")
        
        print(f"处理报告已保存: {report_file}")
    
    def run_full_pipeline(self):
        """运行完整的预处理流程"""
        print("开始GRAMPA数据集预处理流程...")
        
        # 加载数据
        self.load_data()
        
        # 步骤1: 序列合法化
        self.sequence_cleaning()
        
        # 步骤2: 菌株归一化
        self.bacteria_normalization()
        
        # 步骤3: 重复测定处理
        self.handle_duplicates()
        
        # 步骤4: 删失样本处理
        self.handle_censoring()
        
        # 步骤5: 异常值稳健化
        self.winsorize_values()
        
        # 创建序列聚合数据集
        self.create_sequence_aggregated_dataset()
        
        # 步骤6: 数据划分
        self.create_train_val_test_splits()
        
        # 保存所有数据集
        self.save_processed_datasets()
        
        print("\n数据预处理完成！")
        print(f"所有文件已保存到: {self.output_dir}")

def main():
    # 配置参数
    input_file = "/Users/ricardozhao/PycharmProjects/AMP/data/AMP/grampa_merged_dataset.csv"
    output_dir = "/Users/ricardozhao/PycharmProjects/AMP/processed_data"
    
    # 创建预处理器
    preprocessor = GRAMPAPreprocessor(input_file, output_dir)
    
    # 运行完整流程
    preprocessor.run_full_pipeline()

if __name__ == "__main__":
    main()
