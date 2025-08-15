#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze amino acid composition in generated sequences
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def analyze_aa_composition(csv_path):
    """分析氨基酸组成"""
    df = pd.read_csv(csv_path)
    
    print(f"总序列数: {len(df)}")
    print(f"平均长度: {df['length_pred'].mean():.1f}")
    print(f"长度范围: {df['length_pred'].min()}-{df['length_pred'].max()}")
    
    # 合并所有序列
    all_sequences = ''.join(df['aa_seq'].fillna(''))
    total_aa = len(all_sequences)
    
    print(f"总氨基酸数: {total_aa}")
    
    # 统计每种氨基酸
    aa_counts = Counter(all_sequences)
    
    print("\n氨基酸组成分析:")
    print("AA\t数量\t百分比")
    print("-" * 25)
    
    # 按频率排序
    for aa, count in aa_counts.most_common():
        percentage = (count / total_aa) * 100
        print(f"{aa}\t{count}\t{percentage:.2f}%")
    
    # 检查E的比例
    e_ratio = aa_counts.get('E', 0) / total_aa * 100
    print(f"\n*** E的比例: {e_ratio:.2f}% ***")
    
    if e_ratio > 20:
        print("⚠️  E含量异常高！正常蛋白质中E通常占6-7%")
    
    # 检查是否有偏向性
    expected_uniform = 100 / 20  # 5%
    high_bias_aas = []
    for aa, count in aa_counts.items():
        if aa in "ACDEFGHIKLMNPQRSTVWY":
            ratio = (count / total_aa) * 100
            if ratio > expected_uniform * 2:  # 超过均匀分布2倍
                high_bias_aas.append((aa, ratio))
    
    if high_bias_aas:
        print(f"\n高偏向性氨基酸 (>10%):")
        for aa, ratio in sorted(high_bias_aas, key=lambda x: x[1], reverse=True):
            print(f"  {aa}: {ratio:.2f}%")
    
    return aa_counts

def compare_with_natural(aa_counts, total_aa):
    """与天然蛋白质氨基酸组成比较"""
    # 天然蛋白质中氨基酸的大致比例 (%)
    natural_composition = {
        'A': 8.25, 'R': 5.53, 'N': 4.06, 'D': 5.45, 'C': 1.37,
        'Q': 3.93, 'E': 6.75, 'G': 7.07, 'H': 2.27, 'I': 5.96,
        'L': 9.66, 'K': 5.84, 'M': 2.42, 'F': 3.86, 'P': 4.70,
        'S': 6.56, 'T': 5.34, 'W': 1.08, 'Y': 2.92, 'V': 6.87
    }
    
    print("\n与天然蛋白质组成比较:")
    print("AA\t生成\t天然\t差异")
    print("-" * 30)
    
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        generated_pct = (aa_counts.get(aa, 0) / total_aa) * 100
        natural_pct = natural_composition.get(aa, 0)
        diff = generated_pct - natural_pct
        
        status = ""
        if abs(diff) > 5:
            status = " ⚠️" if diff > 0 else " ⬇️"
        
        print(f"{aa}\t{generated_pct:.1f}%\t{natural_pct:.1f}%\t{diff:+.1f}%{status}")

def main():
    csv_path = "decoded_fixed_v2.csv"
    
    print("=== 氨基酸组成分析 ===")
    aa_counts = analyze_aa_composition(csv_path)
    
    total_aa = sum(aa_counts.values())
    compare_with_natural(aa_counts, total_aa)
    
    # 分析可能的原因
    print("\n=== 可能原因分析 ===")
    e_ratio = aa_counts.get('E', 0) / total_aa * 100
    
    if e_ratio > 15:
        print("🔍 E含量过高的可能原因:")
        print("1. 模型偏向生成E token (id=9)")
        print("2. 温度设置可能不够高，导致采样不够随机")
        print("3. embedding特征可能有偏向性")
        print("4. 训练数据中E含量较高")
        
        print("\n💡 建议解决方案:")
        print("1. 提高temperature (如1.2)")
        print("2. 使用更多样化的采样策略")
        print("3. 检查原始训练数据的氨基酸分布")
        print("4. 考虑添加重复惩罚机制")

if __name__ == "__main__":
    main()
