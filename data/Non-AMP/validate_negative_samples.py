#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证负样本质量的脚本
检查负样本的统计信息、长度分布、重复性等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import os
import sys

def check_file_exists(file_path, file_type):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        print(f"错误: {file_type}文件 {file_path} 不存在")
        return False
    return True

def check_file_format(file_path, expected_columns):
    """检查文件格式是否正确"""
    try:
        df = pd.read_csv(file_path, nrows=5)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"错误: 文件 {file_path} 缺少必需的列: {missing_cols}")
            return False
        return True
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return False

def validate_negative_samples(negative_file, positive_file):
    """验证负样本质量"""
    print("正在验证负样本质量...")
    
    # 读取负样本
    print(f"正在读取负样本文件: {negative_file}")
    neg_df = pd.read_csv(negative_file)
    negative_sequences = neg_df['sequence'].tolist()
    
    # 读取正样本
    print(f"正在读取正样本文件: {positive_file}")
    pos_df = pd.read_csv(positive_file)
    positive_sequences = pos_df['sequence'].tolist()
    
    print(f"负样本总数: {len(negative_sequences)}")
    print(f"正样本总数: {len(positive_sequences)}")
    
    # 检查负样本长度分布
    neg_lengths = [len(seq) for seq in negative_sequences]
    pos_lengths = [len(seq) for seq in positive_sequences]
    
    print(f"\n长度统计:")
    print(f"负样本长度范围: {min(neg_lengths)} - {max(neg_lengths)}")
    print(f"负样本平均长度: {np.mean(neg_lengths):.2f}")
    print(f"负样本中位数长度: {np.median(neg_lengths):.2f}")
    print(f"正样本长度范围: {min(pos_lengths)} - {max(pos_lengths)}")
    print(f"正样本平均长度: {np.mean(pos_lengths):.2f}")
    print(f"正样本中位数长度: {np.median(pos_lengths):.2f}")
    
    # 检查重复性
    print(f"\n重复性检查:")
    neg_set = set(negative_sequences)
    pos_set = set(positive_sequences)
    
    # 检查负样本内部重复
    neg_duplicates = len(negative_sequences) - len(neg_set)
    print(f"负样本内部重复数: {neg_duplicates}")
    
    # 检查与正样本的重叠
    overlap = neg_set.intersection(pos_set)
    print(f"与正样本重叠数: {len(overlap)}")
    
    if len(overlap) > 0:
        print("警告: 发现与正样本重叠的序列!")
        print("重叠序列示例:")
        for i, seq in enumerate(list(overlap)[:5]):
            print(f"  {i+1}. {seq}")
    
    # 检查氨基酸组成
    print(f"\n氨基酸组成分析:")
    all_neg_aa = ''.join(negative_sequences)
    all_pos_aa = ''.join(positive_sequences)
    
    neg_aa_counter = Counter(all_neg_aa)
    pos_aa_counter = Counter(all_pos_aa)
    
    print("负样本氨基酸频率 (前10):")
    for aa, count in neg_aa_counter.most_common(10):
        freq = count / len(all_neg_aa) * 100
        print(f"  {aa}: {freq:.2f}%")
    
    print("正样本氨基酸频率 (前10):")
    for aa, count in pos_aa_counter.most_common(10):
        freq = count / len(all_pos_aa) * 100
        print(f"  {aa}: {freq:.2f}%")
    
    # 绘制对比图表
    create_comparison_plots(neg_lengths, pos_lengths, neg_aa_counter, pos_aa_counter)
    
    return {
        'negative_count': len(negative_sequences),
        'positive_count': len(positive_sequences),
        'negative_lengths': neg_lengths,
        'positive_lengths': pos_lengths,
        'overlap_count': len(overlap),
        'internal_duplicates': neg_duplicates
    }

def create_comparison_plots(neg_lengths, pos_lengths, neg_aa_counter, pos_aa_counter):
    """创建对比图表"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 长度分布对比
    axes[0, 0].hist(neg_lengths, bins=50, alpha=0.7, color='red', label='Negative', density=True)
    axes[0, 0].hist(pos_lengths, bins=50, alpha=0.7, color='blue', label='Positive', density=True)
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Length Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 长度箱线图对比
    axes[0, 1].boxplot([neg_lengths, pos_lengths], tick_labels=['Negative', 'Positive'])
    axes[0, 1].set_ylabel('Sequence Length')
    axes[0, 1].set_title('Length Boxplot Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 氨基酸频率对比
    aa_list = list(set(list(neg_aa_counter.keys()) + list(pos_aa_counter.keys())))
    neg_freqs = [neg_aa_counter.get(aa, 0) / sum(neg_aa_counter.values()) * 100 for aa in aa_list]
    pos_freqs = [pos_aa_counter.get(aa, 0) / sum(pos_aa_counter.values()) * 100 for aa in aa_list]
    
    x = np.arange(len(aa_list))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, neg_freqs, width, label='Negative', alpha=0.7, color='red')
    axes[1, 0].bar(x + width/2, pos_freqs, width, label='Positive', alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Amino Acid')
    axes[1, 0].set_ylabel('Frequency (%)')
    axes[1, 0].set_title('Amino Acid Frequency Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(aa_list, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 长度分布密度图
    axes[1, 1].hist(neg_lengths, bins=50, alpha=0.5, color='red', label='Negative', density=True, histtype='step')
    axes[1, 1].hist(pos_lengths, bins=50, alpha=0.5, color='blue', label='Positive', density=True, histtype='step')
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Length Distribution Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('negative_positive_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图表已保存为 negative_positive_comparison.png")

def main():
    """主函数"""
    negative_file = "Non_AMP_UniProtKB.csv"
    positive_file = "../AMP/final_AMP.csv"
    
    print("="*60)
    print("负样本质量验证工具")
    print("="*60)
    
    # 检查文件是否存在
    if not check_file_exists(negative_file, "负样本"):
        return
    
    if not check_file_exists(positive_file, "正样本"):
        return
    
    # 检查文件格式
    if not check_file_format(negative_file, ['sequence', 'source']):
        print("负样本文件格式不正确")
        return
        
    if not check_file_format(positive_file, ['sequence', 'source']):
        print("正样本文件格式不正确")
        return
    
    print("文件检查通过，开始验证...")
    print("-" * 40)
    
    try:
        # 验证负样本
        results = validate_negative_samples(negative_file, positive_file)
        
        # 输出总结
        print(f"\n{'='*50}")
        print("验证总结:")
        print(f"{'='*50}")
        print(f"负样本总数: {results['negative_count']}")
        print(f"正样本总数: {results['positive_count']}")
        print(f"负样本内部重复: {results['internal_duplicates']}")
        print(f"与正样本重叠: {results['overlap_count']}")
        
        if results['overlap_count'] == 0 and results['internal_duplicates'] == 0:
            print("✓ 负样本质量良好: 无重复，无与正样本重叠")
        else:
            print("⚠ 负样本存在问题，需要检查")
        
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
