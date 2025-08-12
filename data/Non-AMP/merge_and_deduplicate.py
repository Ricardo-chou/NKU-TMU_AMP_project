#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并Non-AMPs.fa和Non_AMP_UniProtKB.csv文件并去重
"""

import pandas as pd

def count_fasta_sequences(fasta_file):
    """统计FASTA文件中的序列数"""
    count = 0
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def extract_sequences_from_fasta(fasta_file):
    """从FASTA文件中提取序列"""
    sequences = []
    with open(fasta_file, 'r') as f:
        current_seq = ""
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq.strip())
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq.strip())
    return sequences

def main():
    # 文件路径
    fasta_file = "Non-AMPs.fa"
    csv_file = "Non_AMP_UniProtKB.csv"
    
    print("开始处理文件...")
    
    # 统计FASTA文件序列数
    fasta_count = count_fasta_sequences(fasta_file)
    print(f"Non-AMPs.fa 文件包含 {fasta_count} 条序列")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    csv_count = len(df)
    print(f"Non_AMP_UniProtKB.csv 文件包含 {csv_count} 条序列")
    
    # 从FASTA文件提取序列
    print("正在从FASTA文件提取序列...")
    fasta_sequences = extract_sequences_from_fasta(fasta_file)
    
    # 从CSV文件获取序列
    csv_sequences = df['sequence'].tolist()
    
    # 合并所有序列
    all_sequences = fasta_sequences + csv_sequences
    print(f"合并后总序列数: {len(all_sequences)}")
    
    # 去重
    unique_sequences = list(set(all_sequences))
    print(f"去重后序列数: {len(unique_sequences)}")
    
    # 创建新的DataFrame
    result_df = pd.DataFrame({
        'sequence': unique_sequences,
        'source': ['FASTA' if seq in fasta_sequences else 'CSV' for seq in unique_sequences]
    })
    
    # 保存结果
    output_file = "final_non_amps.csv"
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 统计来源分布
    source_counts = result_df['source'].value_counts()
    print("\n来源分布:")
    print(f"来自FASTA文件: {source_counts.get('FASTA', 0)}")
    print(f"来自CSV文件: {source_counts.get('CSV', 0)}")

if __name__ == "__main__":
    main()
