#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从UniProtKB FASTA文件生成负样本集的脚本
分析正样本长度分布，随机截取片段作为负样本
"""

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import random
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_positive_lengths(positive_file):
    """分析正样本的长度分布"""
    print("正在分析正样本长度分布...")
    
    # 读取正样本
    df = pd.read_csv(positive_file)
    sequences = df['sequence'].tolist()
    
    # 计算长度分布
    lengths = [len(seq) for seq in sequences]
    length_counter = Counter(lengths)
    
    print(f"正样本总数: {len(sequences)}")
    print(f"长度范围: {min(lengths)} - {max(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    
    # 绘制长度分布图
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Positive Sample Length Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_lengths = sorted(length_counter.items())
    x, y = zip(*sorted_lengths)
    plt.plot(x, y, marker='o', markersize=3, linewidth=1)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Positive Sample Length Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positive_length_distribution.png', dpi=300, bbox_inches='tight')
    print("长度分布图已保存为 positive_length_distribution.png")
    # plt.show()  # 注释掉以避免在服务器环境中显示问题
    
    return sequences, lengths, length_counter

def generate_negative_samples(fasta_file, positive_sequences, length_distribution, 
                            num_samples=10000, min_length=5, max_length=200):
    """从FASTA文件生成负样本"""
    print(f"正在从 {fasta_file} 生成负样本...")
    
    # 创建正样本集合用于去重
    positive_set = set(positive_sequences)
    
    negative_samples = []
    attempts = 0
    max_attempts = num_samples * 100  # 最大尝试次数
    
    print("正在解析FASTA文件...")
    
    # 读取FASTA文件
    with open(fasta_file, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if len(negative_samples) >= num_samples:
                break
                
            seq_str = str(record.seq)
            
            # 跳过太短的序列
            if len(seq_str) < min_length:
                continue
            
            # 随机截取片段
            for _ in range(10):  # 每个蛋白质尝试10次
                if len(negative_samples) >= num_samples:
                    break
                    
                # 随机选择长度（基于正样本分布）
                target_length = random.choices(
                    list(length_distribution.keys()),
                    weights=list(length_distribution.values())
                )[0]
                
                # 确保目标长度在合理范围内
                target_length = max(min_length, min(max_length, target_length))
                
                # 随机选择起始位置
                if len(seq_str) <= target_length:
                    continue
                    
                start_pos = random.randint(0, len(seq_str) - target_length)
                fragment = seq_str[start_pos:start_pos + target_length]
                
                # 检查是否为重复序列
                if fragment not in positive_set and fragment not in negative_samples:
                    negative_samples.append(fragment)
                    if len(negative_samples) % 5000 == 0:
                        print(f"已生成 {len(negative_samples)} 个负样本 (目标: {num_samples})")
                
                attempts += 1
                if attempts >= max_attempts:
                    print(f"达到最大尝试次数 {max_attempts}，停止生成")
                    break
    
    print(f"成功生成 {len(negative_samples)} 个负样本")
    return negative_samples

def save_negative_samples(negative_samples, output_file):
    """保存负样本到CSV文件"""
    print(f"正在保存负样本到 {output_file}...")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'sequence': negative_samples,
        'source': 'UniProtKB_Negative'
    })
    
    # 保存到CSV
    df.to_csv(output_file, index=False)
    print(f"负样本已保存到 {output_file}")
    
    # 显示负样本统计信息
    lengths = [len(seq) for seq in negative_samples]
    print(f"负样本长度统计:")
    print(f"  总数: {len(negative_samples)}")
    print(f"  长度范围: {min(lengths)} - {max(lengths)}")
    print(f"  平均长度: {np.mean(lengths):.2f}")
    print(f"  中位数长度: {np.median(lengths):.2f}")

def main():
    """主函数"""
    # 文件路径
    positive_file = "../AMP/final_AMP.csv"
    fasta_file = "uniprot_sprot.fasta"
    output_file = "Non_AMP_UniProtKB.csv"
    
    # 检查文件是否存在
    if not os.path.exists(positive_file):
        print(f"错误: 正样本文件 {positive_file} 不存在")
        return
    
    if not os.path.exists(fasta_file):
        print(f"错误: FASTA文件 {fasta_file} 不存在")
        return
    
    # 分析正样本长度分布
    positive_sequences, lengths, length_distribution = analyze_positive_lengths(positive_file)
    
    # 生成负样本
    negative_samples = generate_negative_samples(
        fasta_file, 
        positive_sequences, 
        length_distribution,
        num_samples=50000,  # 生成50000个负样本
        min_length=5,
        max_length=200
    )
    
    # 保存负样本
    save_negative_samples(negative_samples, output_file)
    
    print("负样本生成完成！")

if __name__ == "__main__":
    main()
