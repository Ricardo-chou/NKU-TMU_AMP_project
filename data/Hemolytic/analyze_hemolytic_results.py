#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
溶血性标签处理结果分析脚本

分析处理后的溶血性数据，提供详细的统计信息和质量评估

作者: AI Assistant
日期: 2025-01-15
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def analyze_hemolytic_results(file_path):
    """分析溶血性处理结果"""
    
    # 读取数据
    df = pd.read_csv(file_path)
    print(f"数据总量: {len(df)} 条记录")
    
    # 基本统计
    print("\n=== 基本统计信息 ===")
    print(f"高毒性 (1): {len(df[df['hemolytic_binary'] == 1])} ({len(df[df['hemolytic_binary'] == 1])/len(df)*100:.1f}%)")
    print(f"低毒性 (0): {len(df[df['hemolytic_binary'] == 0])} ({len(df[df['hemolytic_binary'] == 0])/len(df)*100:.1f}%)")
    
    # 处理类型统计
    print("\n=== 处理类型统计 ===")
    process_counts = df['process_type'].value_counts()
    for ptype, count in process_counts.items():
        print(f"{ptype}: {count} ({count/len(df)*100:.1f}%)")
    
    # 各类型的毒性分布
    print("\n=== 各处理类型的毒性分布 ===")
    for ptype in process_counts.index:
        subset = df[df['process_type'] == ptype]
        high_toxic = len(subset[subset['hemolytic_binary'] == 1])
        low_toxic = len(subset[subset['hemolytic_binary'] == 0])
        print(f"{ptype}: 高毒 {high_toxic} ({high_toxic/len(subset)*100:.1f}%), 低毒 {low_toxic} ({low_toxic/len(subset)*100:.1f}%)")
    
    # 分析无法解析的数据
    unparseable = df[df['process_type'] == 'unparseable']
    if len(unparseable) > 0:
        print(f"\n=== 无法解析数据分析 (共{len(unparseable)}条) ===")
        unique_patterns = unparseable['Hemolytic Activity'].value_counts().head(10)
        for pattern, count in unique_patterns.items():
            print(f"{count}x: {pattern}")
    
    # 数据质量评估
    print(f"\n=== 数据质量评估 ===")
    successfully_parsed = len(df) - len(unparseable)
    print(f"成功解析率: {successfully_parsed/len(df)*100:.1f}%")
    print(f"数据完整性: 良好" if successfully_parsed/len(df) > 0.85 else "需要改进")
    
    # 保存分析结果
    analysis_results = {
        'total_records': len(df),
        'high_toxic_count': len(df[df['hemolytic_binary'] == 1]),
        'low_toxic_count': len(df[df['hemolytic_binary'] == 0]),
        'high_toxic_percentage': len(df[df['hemolytic_binary'] == 1])/len(df)*100,
        'low_toxic_percentage': len(df[df['hemolytic_binary'] == 0])/len(df)*100,
        'successfully_parsed_rate': successfully_parsed/len(df)*100,
        'unparseable_count': len(unparseable),
        'process_type_distribution': process_counts.to_dict()
    }
    
    return analysis_results, df

def create_summary_report(df, output_file):
    """创建处理结果摘要报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 溶血性标签二分类处理结果报告\n\n")
        f.write(f"处理日期: 2025-01-15\n")
        f.write(f"数据来源: CAMP溶血性数据\n")
        f.write(f"总记录数: {len(df)}\n\n")
        
        f.write("## 处理规则总结\n\n")
        f.write("### 高毒性 (1) 判定规则:\n")
        f.write("1. **半效应浓度类**: LC50/HC50/MHC/HL50/IH50/EC50/LD50 ≤ 50 μM\n")
        f.write("2. **百分比溶血类**: 溶血率 ≥20% 且浓度 ≤64 μM\n")
        f.write("3. **描述性高毒**: 包含'has hemolytic activity', 'exhibits hemolysis'等关键词\n\n")
        
        f.write("### 低毒性 (0) 判定规则:\n")
        f.write("1. **显式非溶血性**: 包含'non-hemolytic', 'no hemolytic'等关键词\n")
        f.write("2. **低毒性描述**: 包含'low hemolytic activity'等关键词\n")
        f.write("3. **其他情况**: 不满足高毒性条件的数据\n\n")
        
        f.write("## 处理结果统计\n\n")
        f.write(f"- **高毒性样本**: {len(df[df['hemolytic_binary'] == 1])} 条 ({len(df[df['hemolytic_binary'] == 1])/len(df)*100:.1f}%)\n")
        f.write(f"- **低毒性样本**: {len(df[df['hemolytic_binary'] == 0])} 条 ({len(df[df['hemolytic_binary'] == 0])/len(df)*100:.1f}%)\n\n")
        
        f.write("## 处理类型分布\n\n")
        process_counts = df['process_type'].value_counts()
        for ptype, count in process_counts.items():
            f.write(f"- **{ptype}**: {count} 条 ({count/len(df)*100:.1f}%)\n")
        
        f.write(f"\n## 数据质量\n\n")
        unparseable = df[df['process_type'] == 'unparseable']
        successfully_parsed = len(df) - len(unparseable)
        f.write(f"- **成功解析率**: {successfully_parsed/len(df)*100:.1f}%\n")
        f.write(f"- **无法解析**: {len(unparseable)} 条 ({len(unparseable)/len(df)*100:.1f}%)\n")
        
        f.write("\n## 使用建议\n\n")
        f.write("1. **机器学习应用**: 该二分类标签可直接用于溶血性预测模型训练\n")
        f.write("2. **数据筛选**: 可根据process_type字段筛选特定类型的数据进行分析\n")
        f.write("3. **质量控制**: 建议人工检查process_type为'unparseable'的数据\n")
        f.write("4. **进一步改进**: 对于μg/mL单位的数据，可结合分子量信息进行更精确的分类\n")

def main():
    """主函数"""
    input_file = "/Users/ricardozhao/PycharmProjects/AMP/data/Hemolytic/CAMP_hemolysis_processed.csv"
    report_file = "/Users/ricardozhao/PycharmProjects/AMP/data/Hemolytic/hemolytic_processing_report.md"
    
    print("分析溶血性处理结果...")
    results, df = analyze_hemolytic_results(input_file)
    
    print("\n生成摘要报告...")
    create_summary_report(df, report_file)
    print(f"报告已保存到: {report_file}")
    
    # 显示一些具体的处理示例
    print("\n=== 处理示例验证 ===")
    
    # 高毒性示例
    high_toxic_examples = df[df['hemolytic_binary'] == 1].head(5)
    print("\n高毒性样本示例:")
    for idx, row in high_toxic_examples.iterrows():
        print(f"  {row['AMP ID']}: {row['Hemolytic Activity']} -> {row['process_type']}")
    
    # 低毒性示例
    low_toxic_examples = df[df['hemolytic_binary'] == 0].head(5)
    print("\n低毒性样本示例:")
    for idx, row in low_toxic_examples.iterrows():
        print(f"  {row['AMP ID']}: {row['Hemolytic Activity']} -> {row['process_type']}")

if __name__ == "__main__":
    main()
