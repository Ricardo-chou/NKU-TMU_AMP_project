#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
序列筛选脚本
合并两个CSV文件并应用严格的筛选规则
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

def calculate_net_charge(sequence, ph=7.0):
    """
    计算序列在指定pH下的净电荷
    使用简化的pKa模型
    """
    # 氨基酸pKa值 (简化模型)
    pka_values = {
        'K': 10.5,  # Lys
        'R': 12.5,  # Arg  
        'H': 6.0,   # His
        'D': 3.9,   # Asp
        'E': 4.3,   # Glu
        'C': 8.3,   # Cys
        'Y': 10.1,  # Tyr
    }
    
    # N端和C端
    n_term_pka = 9.6
    c_term_pka = 2.3
    
    net_charge = 0.0
    
    # N端贡献 (正电荷)
    net_charge += 1 / (1 + 10**(ph - n_term_pka))
    
    # C端贡献 (负电荷)
    net_charge -= 1 / (1 + 10**(c_term_pka - ph))
    
    # 各氨基酸贡献
    for aa in sequence:
        if aa in pka_values:
            pka = pka_values[aa]
            if aa in ['K', 'R', 'H']:  # 碱性氨基酸 (正电荷)
                net_charge += 1 / (1 + 10**(ph - pka))
            else:  # 酸性氨基酸 (负电荷)
                net_charge -= 1 / (1 + 10**(pka - ph))
    
    return net_charge

def has_long_repeats(sequence, max_repeat=10):
    """
    检查序列是否有超过max_repeat的连续重复氨基酸
    """
    current_aa = sequence[0] if sequence else ''
    count = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_aa:
            count += 1
            if count > max_repeat:
                return True
        else:
            current_aa = sequence[i]
            count = 1
    
    return False

def calculate_kr_ratio(sequence):
    """
    计算K+R占比
    """
    if not sequence:
        return 0.0
    
    k_count = sequence.count('K')
    r_count = sequence.count('R')
    total_length = len(sequence)
    
    return (k_count + r_count) / total_length

def load_and_merge_data():
    """
    加载并合并两个CSV文件
    """
    print("正在加载数据文件...")
    
    # 读取两个文件
    df1 = pd.read_csv('/root/NKU-TMU_AMP_project/decode/full_data/decoded_optimized.csv')
    df2 = pd.read_csv('/root/NKU-TMU_AMP_project/decode/full_data/decoded_optimized2.csv')
    
    print(f"文件1包含 {len(df1)} 条序列")
    print(f"文件2包含 {len(df2)} 条序列")
    
    # 合并数据
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"合并后共 {len(merged_df)} 条序列")
    
    return merged_df

def load_known_sequences():
    """
    加载已知的AMP序列
    """
    try:
        final_amp_df = pd.read_csv('/root/NKU-TMU_AMP_project/data/AMP/final_AMP.csv')
        # 假设序列列名为'sequence'，如果不是需要调整
        if 'sequence' in final_amp_df.columns:
            known_sequences = set(final_amp_df['sequence'].dropna())
        elif 'Sequence' in final_amp_df.columns:
            known_sequences = set(final_amp_df['Sequence'].dropna())
        else:
            # 取第一列作为序列列
            known_sequences = set(final_amp_df.iloc[:, 0].dropna())
        
        print(f"加载了 {len(known_sequences)} 条已知AMP序列")
        return known_sequences
    except Exception as e:
        print(f"加载已知序列时出错: {e}")
        return set()

def apply_filters(df, known_sequences=None):
    """
    应用所有筛选规则
    """
    print("\n开始应用筛选规则...")
    original_count = len(df)
    
    # 确保有序列列
    sequence_col = None
    for col in ['aa_seq', 'sequence', 'Sequence', 'seq']:
        if col in df.columns:
            sequence_col = col
            break
    
    if sequence_col is None:
        # 假设第一列是序列
        sequence_col = df.columns[0]
    
    print(f"使用列 '{sequence_col}' 作为序列数据")
    
    # 1. 去除空序列
    df = df.dropna(subset=[sequence_col])
    df = df[df[sequence_col].str.len() > 0]
    print(f"去除空序列后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 2. 去除重复序列
    df = df.drop_duplicates(subset=[sequence_col])
    print(f"去除重复序列后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 3. 去除已知序列
    if known_sequences:
        mask = ~df[sequence_col].isin(known_sequences)
        df = df[mask]
        print(f"去除已知序列后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 4. 连续重复筛选
    print("应用连续重复筛选...")
    mask = ~df[sequence_col].apply(lambda x: has_long_repeats(x, max_repeat=10))
    df = df[mask]
    print(f"连续重复筛选后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 5. 净电荷筛选
    print("计算净电荷...")
    net_charges = df[sequence_col].apply(calculate_net_charge)
    mask = net_charges > 0
    df = df[mask]
    print(f"净电荷>0筛选后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 6. K+R占比筛选
    print("计算K+R占比...")
    kr_ratios = df[sequence_col].apply(calculate_kr_ratio)
    mask = kr_ratios <= 0.4
    df = df[mask]
    print(f"K+R占比≤40%筛选后: {len(df)} 条 (减少 {original_count - len(df)} 条)")
    
    # 添加统计信息列
    df['net_charge'] = df[sequence_col].apply(calculate_net_charge)
    df['kr_ratio'] = df[sequence_col].apply(calculate_kr_ratio)
    df['length'] = df[sequence_col].str.len()
    
    return df

def main():
    """
    主函数
    """
    print("=== AMP序列筛选程序 ===")
    
    # 1. 加载和合并数据
    merged_df = load_and_merge_data()
    
    # 2. 加载已知序列
    known_sequences = load_known_sequences()
    
    # 3. 应用筛选规则
    filtered_df = apply_filters(merged_df, known_sequences)
    
    # 4. 保存结果
    output_path = '/root/NKU-TMU_AMP_project/decode/filtered_candidate_sequences.csv'
    filtered_df.to_csv(output_path, index=False)
    
    print(f"\n筛选完成！")
    print(f"原始序列数: {len(merged_df)}")
    print(f"筛选后序列数: {len(filtered_df)}")
    print(f"筛选率: {len(filtered_df)/len(merged_df)*100:.1f}%")
    print(f"结果已保存到: {output_path}")
    
    # 显示一些统计信息
    if len(filtered_df) > 0:
        sequence_col = None
        for col in ['aa_seq', 'sequence', 'Sequence', 'seq']:
            if col in filtered_df.columns:
                sequence_col = col
                break
        if sequence_col is None:
            sequence_col = filtered_df.columns[0]
            
        print(f"\n统计信息:")
        print(f"序列长度范围: {filtered_df['length'].min()}-{filtered_df['length'].max()}")
        print(f"平均长度: {filtered_df['length'].mean():.1f}")
        print(f"净电荷范围: {filtered_df['net_charge'].min():.2f}-{filtered_df['net_charge'].max():.2f}")
        print(f"平均净电荷: {filtered_df['net_charge'].mean():.2f}")
        print(f"K+R占比范围: {filtered_df['kr_ratio'].min():.3f}-{filtered_df['kr_ratio'].max():.3f}")
        print(f"平均K+R占比: {filtered_df['kr_ratio'].mean():.3f}")

if __name__ == "__main__":
    main()
