#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取AMPs.fa和positive_amp_dataset.csv中的sequence并合并去重
输出final_AMP.csv文件
"""

import pandas as pd
import re

def extract_fasta_sequences(fasta_file):
    """从FASTA文件中提取序列"""
    sequences = set()
    
    with open(fasta_file, 'r', encoding='utf-8') as f:
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_sequence:
                    sequences.add(current_sequence.upper())
                current_sequence = ""
            else:
                # 累积序列
                current_sequence += line
        
        # 保存最后一个序列
        if current_sequence:
            sequences.add(current_sequence.upper())
    
    return sequences

def extract_csv_sequences(csv_file):
    """从CSV文件中提取sequence列"""
    try:
        df = pd.read_csv(csv_file)
        if 'sequence' in df.columns:
            sequences = set(df['sequence'].dropna().astype(str).str.upper())
            return sequences
        else:
            print(f"警告: {csv_file} 中没有找到'sequence'列")
            return set()
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return set()

def main():
    # 文件路径
    fasta_file = "data/AMP/AMPs.fa"
    csv_file = "data/AMP/positive_amp_dataset.csv"
    output_file = "final_AMP.csv"
    
    print("开始提取序列...")
    
    # 从FASTA文件提取序列
    print(f"正在从 {fasta_file} 提取序列...")
    fasta_sequences = extract_fasta_sequences(fasta_file)
    print(f"从FASTA文件提取到 {len(fasta_sequences)} 个序列")
    
    # 从CSV文件提取序列
    print(f"正在从 {csv_file} 提取序列...")
    csv_sequences = extract_csv_sequences(csv_file)
    print(f"从CSV文件提取到 {len(csv_sequences)} 个序列")
    
    # 合并所有序列并去重
    all_sequences = fasta_sequences.union(csv_sequences)
    print(f"合并去重后共有 {len(all_sequences)} 个唯一序列")
    
    # 创建输出DataFrame
    output_df = pd.DataFrame({
        'sequence': sorted(list(all_sequences)),
        'source': ['FASTA' if seq in fasta_sequences else 'CSV' for seq in sorted(all_sequences)]
    })
    
    # 保存到CSV文件
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"结果已保存到 {output_file}")
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"FASTA文件独有序列: {len(fasta_sequences - csv_sequences)}")
    print(f"CSV文件独有序列: {len(csv_sequences - fasta_sequences)}")
    print(f"两个文件共有序列: {len(fasta_sequences.intersection(csv_sequences))}")
    
    # 显示前几个序列示例
    print(f"\n前5个序列示例:")
    for i, seq in enumerate(sorted(all_sequences)[:5]):
        print(f"{i+1}. {seq}")

if __name__ == "__main__":
    main()
