#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并CAMP和grampa数据集的脚本
统一单位并去重
"""

import pandas as pd
import numpy as np
import re
import math

# 标准氨基酸字母表
STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

def is_valid_sequence(sequence):
    """
    验证序列是否只包含标准氨基酸
    """
    if pd.isna(sequence) or not isinstance(sequence, str):
        return False
    
    # 检查序列长度
    if len(sequence) < 5 or len(sequence) > 100:
        return False
    
    # 检查是否只包含标准氨基酸
    sequence_upper = sequence.upper()
    return all(aa in STANDARD_AMINO_ACIDS for aa in sequence_upper)

def clean_and_filter_dataset(df):
    """
    清洗和筛选数据集
    """
    print("开始数据清洗...")
    initial_count = len(df)
    
    # 1. 仅保留由标准氨基酸组成的肽序列
    print("1. 筛选标准氨基酸序列...")
    df = df[df['sequence'].apply(is_valid_sequence)]
    print(f"   标准氨基酸序列筛选后: {len(df)}行")
    
    # 2. 筛选序列长度在5-100个氨基酸之间
    print("2. 筛选序列长度5-100...")
    df = df[df['sequence'].str.len().between(5, 100)]
    print(f"   长度筛选后: {len(df)}行")
    
    # 3. 去除重复序列以保证样本唯一性
    print("3. 去除重复序列...")
    df = df.drop_duplicates(subset=['sequence'], keep='first')
    print(f"   去重后: {len(df)}行")
    
    # 4. 对多条记录的同一肽序列取最小MIC作为代表值
    print("4. 对同一序列取最小MIC值...")
    # 按序列分组，取最小value值
    df = df.groupby('sequence').agg({
        'bacterium': lambda x: '; '.join(set(x.dropna())),  # 合并细菌名称
        'unit': 'first',  # 单位应该都是uM
        'value': 'min',   # 取最小MIC值（最高活性）
        'censor': lambda x: x.iloc[x.argmin()] if len(x) > 0 else None,  # 对应最小值的censor
        'database': lambda x: '; '.join(set(x))  # 合并数据库来源
    }).reset_index()
    
    print(f"   序列聚合后: {len(df)}行")
    
    # 5. 移除无效的MIC值
    print("5. 移除无效MIC值...")
    df = df.dropna(subset=['value'])
    print(f"   移除无效值后: {len(df)}行")
    
    # 按序列长度排序
    df = df.sort_values('sequence')
    
    final_count = len(df)
    print(f"\n数据清洗完成:")
    print(f"   原始数据: {initial_count}行")
    print(f"   清洗后: {final_count}行")
    print(f"   减少: {initial_count - final_count}行")
    
    return df

def convert_to_uM_log(value_str, unit, sequence):
    """
    将不同单位转换为uM的对数值，同时返回censor信息
    返回: (log10_uM, censor)
    """
    if pd.isna(value_str) or value_str == '':
        return np.nan, None
    
    # 处理特殊值和范围
    censor = None
    if isinstance(value_str, str):
        value_str = value_str.strip()
        # 处理不等号
        if value_str.startswith('>'):
            censor = '>'
            value_str = value_str[1:]
        elif value_str.startswith('<'):
            censor = '<'
            value_str = value_str[1:]
        elif value_str.startswith('≤'):
            censor = '≤'
            value_str = value_str[1:]
        elif value_str.startswith('≥'):
            censor = '≥'
            value_str = value_str[1:]
        elif value_str.startswith('='):
            value_str = value_str[1:]
        
        # 处理范围值 (如 "16-32")
        if '-' in value_str and not value_str.startswith('-'):
            try:
                parts = value_str.split('-')
                if len(parts) == 2:
                    low, high = float(parts[0]), float(parts[1])
                    value = (low + high) / 2  # 取中点
                else:
                    value = float(value_str)
            except ValueError:
                return np.nan, None
        else:
            # 移除逗号
            value_str = value_str.replace(',', '')
            try:
                value = float(value_str)
            except ValueError:
                return np.nan, None
    else:
        value = float(value_str)
    
    # 单位转换 - 注意这里不做lower()，保持原始大小写
    unit = str(unit).strip()
    
    # 计算分子量 (假设平均氨基酸分子量为110 Da)
    if pd.notna(sequence) and isinstance(sequence, str):
        molecular_weight_Da = len(sequence) * 110
    else:
        molecular_weight_Da = 1500  # 默认值
    
    # 兼容多种微符号：μ (Greek mu), µ (micro sign), micro
    if any(symbol in unit for symbol in ['microg/ml', 'μg/ml', 'ug/ml', 'µg/ml']):
        # 转换为 μM: μM = (mg/L) / (g/mol) * 1e6
        # microg/ml = mg/L
        mg_per_L = value
        uM = (mg_per_L / 1000) / (molecular_weight_Da / 1) * 1e6
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif 'mg/ml' in unit:
        # mg/ml = 1000 * mg/L
        mg_per_L = value * 1000
        uM = (mg_per_L / 1000) / (molecular_weight_Da / 1) * 1e6
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif 'g/ml' in unit:
        # g/ml = 1000000 * mg/L
        mg_per_L = value * 1000000
        uM = (mg_per_L / 1000) / (molecular_weight_Da / 1) * 1e6
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif 'ng/ml' in unit:
        # ng/ml = 0.001 * mg/L
        mg_per_L = value * 0.001
        uM = (mg_per_L / 1000) / (molecular_weight_Da / 1) * 1e6
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif 'pg/ml' in unit:
        # pg/ml = 0.000001 * mg/L
        mg_per_L = value * 0.000001
        uM = (mg_per_L / 1000) / (molecular_weight_Da / 1) * 1e6
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif any(symbol in unit for symbol in ['um', 'μm', 'uM', 'µM']):
        # 已经是uM单位，直接取对数
        return (math.log10(value), censor) if value > 0 else (np.nan, censor)
        
    elif any(symbol in unit for symbol in ['nm', 'nM', 'nµM']):
        # nM = 0.001 * uM
        uM = value * 0.001
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif any(symbol in unit for symbol in ['pm', 'pM', 'pµM']):
        # pM = 0.000001 * uM
        uM = value * 0.000001
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    elif any(symbol in unit for symbol in ['mm', 'mM', 'mµM']):
        # mM = 1000 * uM
        uM = value * 1000
        return (math.log10(uM), censor) if uM > 0 else (np.nan, censor)
        
    else:
        # 未知单位，返回NaN
        return np.nan, None

def clean_bacterium_name(name):
    """
    清理细菌名称，统一格式
    """
    if pd.isna(name) or name == '':
        return ''
    
    name = str(name).strip()
    
    # 移除引号和逗号
    name = name.replace('"', '').replace(',', '').strip()
    
    # 移除常见的菌株标识
    name = re.sub(r'\s+ATCC\s+\d+', '', name)
    name = re.sub(r'\s+[A-Z]+\d+', '', name)
    name = re.sub(r'\s+[A-Z]+\s+\d+', '', name)
    
    # 标准化常见细菌名称
    name_mapping = {
        'S. aureus': 'Staphylococcus aureus',
        'E. coli': 'Escherichia coli',
        'P. aeruginosa': 'Pseudomonas aeruginosa',
        'B. subtilis': 'Bacillus subtilis',
        'C. albicans': 'Candida albicans',
        'S. epidermidis': 'Staphylococcus epidermidis',
        'S. pneumoniae': 'Streptococcus pneumoniae',
        'S. pyogenes': 'Streptococcus pyogenes',
        'L. monocytogenes': 'Listeria monocytogenes',
        'S. typhimurium': 'Salmonella typhimurium',
        'A. baumannii': 'Acinetobacter baumannii',
        'K. pneumoniae': 'Klebsiella pneumoniae',
        'E. faecium': 'Enterococcus faecium',
        'E. faecalis': 'Enterococcus faecalis',
        'S. agalactiae': 'Streptococcus agalactiae',
        'S. mutans': 'Streptococcus mutans',
        'S. sobrinus': 'Streptococcus sobrinus',
        'S. salivarius': 'Streptococcus salivarius',
        'S. sanguinis': 'Streptococcus sanguinis',
        'S. oralis': 'Streptococcus oralis',
        'S. parasanguinis': 'Streptococcus parasanguinis',
        'P. intermedia': 'Prevotella intermedia',
        'P. gingivalis': 'Porphyromonas gingivalis',
        'F. nucleatum': 'Fusobacterium nucleatum',
        'A. israelii': 'Actinomyces israelii',
        'S. enterica': 'Salmonella enterica',
        'A. salmonicida': 'Aeromonas salmonicida',
        'Y. pseudotuberculosis': 'Yersinia pseudotuberculosis',
        'L. mexicana': 'Leishmania mexicana',
        'L. donovani': 'Leishmania donovani',
        'M. tuberculosis': 'Mycobacterium tuberculosis',
        'M. luteus': 'Micrococcus luteus',
        'B. cereus': 'Bacillus cereus',
        'B. megaterium': 'Bacillus megaterium',
        'B. negaterium': 'Bacillus negaterium',
        'B. pyocyaneus': 'Pseudomonas aeruginosa',
        'C. michiganensis': 'Clavibacter michiganensis',
        'R. solani': 'Rhizoctonia solani',
        'E. tarda': 'Edwardsiella tarda',
        'A. johnsonii': 'Acinetobacter johnsonii',
        'B. dysenteriae': 'Shigella dysenteriae'
    }
    
    for short_name, full_name in name_mapping.items():
        if name == short_name:
            return full_name
    
    return name

def merge_datasets():
    """
    合并两个数据集
    """
    print("开始读取数据集...")
    
    # 读取CAMP数据集
    try:
        camp_df = pd.read_csv('data/CAMP_activity_data copy.csv')
        print(f"CAMP数据集读取成功，共{len(camp_df)}行")
    except Exception as e:
        print(f"读取CAMP数据集失败: {e}")
        return
    
    # 读取grampa数据集
    try:
        grampa_df = pd.read_csv('data/grampa.csv')
        print(f"grampa数据集读取成功，共{len(grampa_df)}行")
    except Exception as e:
        print(f"读取grampa数据集失败: {e}")
        return
    
    print("\n开始处理CAMP数据集...")
    
    # 处理CAMP数据集
    camp_processed = camp_df.copy()
    
    # 重命名列以匹配目标格式
    camp_processed = camp_processed.rename(columns={
        '抵抗微生物名': 'bacterium',
        'sequence': 'sequence',
        'unit': 'unit',
        'value': 'value',
        'url_source': 'url_source'
    })
    
    # 添加database列
    camp_processed['database'] = 'CAMP'
    
    # 清理细菌名称
    camp_processed['bacterium'] = camp_processed['bacterium'].apply(clean_bacterium_name)
    
    # 统一单位并转换数值
    conversion_results = camp_processed.apply(
        lambda row: convert_to_uM_log(row['value'], row['unit'], row['sequence']), axis=1
    )
    
    # 分离数值和censor信息
    camp_processed['value_converted'] = [result[0] if isinstance(result, tuple) else result for result in conversion_results]
    camp_processed['censor'] = [result[1] if isinstance(result, tuple) else None for result in conversion_results]
    
    # 更新unit为统一单位
    camp_processed['unit'] = 'uM'
    
    # 选择需要的列
    camp_processed = camp_processed[['bacterium', 'sequence', 'unit', 'value_converted', 'censor', 'database']]
    camp_processed = camp_processed.rename(columns={'value_converted': 'value'})
    
    print(f"CAMP数据集处理完成，保留{len(camp_processed)}行")
    
    print("\n开始处理grampa数据集...")
    
    # 处理grampa数据集
    grampa_processed = grampa_df.copy()
    
    # 重命名列以匹配目标格式
    grampa_processed = grampa_processed.rename(columns={
        'bacterium': 'bacterium',
        'sequence': 'sequence',
        'unit': 'unit',
        'value': 'value',
        'database': 'database'
    })
    
    # 清理细菌名称
    grampa_processed['bacterium'] = grampa_processed['bacterium'].apply(clean_bacterium_name)
    
    # 统一单位并转换数值
    conversion_results = grampa_processed.apply(
        lambda row: convert_to_uM_log(row['value'], row['unit'], row['sequence']), axis=1
    )
    
    # 分离数值和censor信息
    grampa_processed['value_converted'] = [result[0] if isinstance(result, tuple) else result for result in conversion_results]
    grampa_processed['censor'] = [result[1] if isinstance(result, tuple) else None for result in conversion_results]
    
    # 更新unit为统一单位
    grampa_processed['unit'] = 'uM'
    
    # 选择需要的列
    grampa_processed = grampa_processed[['bacterium', 'sequence', 'unit', 'value_converted', 'censor', 'database']]
    grampa_processed = grampa_processed.rename(columns={'value_converted': 'value'})
    
    print(f"grampa数据集处理完成，保留{len(grampa_processed)}行")
    
    print("\n开始合并数据集...")
    
    # 合并数据集
    merged_df = pd.concat([camp_processed, grampa_processed], ignore_index=True)
    print(f"合并后数据集共{len(merged_df)}行")
    
    print("\n开始去重...")
    
    # 去重：基于细菌名称、序列和数据库的组合
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['bacterium', 'sequence', 'database'], keep='first')
    final_count = len(merged_df)
    
    print(f"去重前: {initial_count}行")
    print(f"去重后: {final_count}行")
    print(f"移除重复行: {initial_count - final_count}行")
    
    # 移除空值行
    merged_df = merged_df.dropna(subset=['bacterium', 'sequence', 'value'])
    print(f"移除空值后: {len(merged_df)}行")
    
    # 按细菌名称和序列排序
    merged_df = merged_df.sort_values(['bacterium', 'sequence'])
    
    print("\n保存合并后的数据集...")
    
    # 保存合并后的数据集
    output_file = 'data/merged_amp_dataset.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"合并后的数据集已保存到: {output_file}")
    
    # 显示数据集统计信息
    print("\n=== 合并后数据集统计信息 ===")
    print(f"总行数: {len(merged_df)}")
    print(f"唯一细菌数: {merged_df['bacterium'].nunique()}")
    print(f"唯一序列数: {merged_df['sequence'].nunique()}")
    print(f"数据库分布:")
    print(merged_df['database'].value_counts())
    
    print("\n开始数据清洗...")
    
    # 数据清洗和筛选
    cleaned_df = clean_and_filter_dataset(merged_df)
    
    # 保存清洗后的正样本AMP数据集
    positive_amp_file = 'data/positive_amp_dataset.csv'
    cleaned_df.to_csv(positive_amp_file, index=False, encoding='utf-8')
    print(f"\n正样本AMP数据集已保存到: {positive_amp_file}")
    
    # 显示清洗后数据集统计信息
    print("\n=== 正样本AMP数据集统计信息 ===")
    print(f"总行数: {len(cleaned_df)}")
    print(f"唯一序列数: {cleaned_df['sequence'].nunique()}")
    print(f"序列长度分布:")
    length_stats = cleaned_df['sequence'].str.len().describe()
    print(length_stats)
    
    # 显示前几行数据
    print("\n=== 前5行数据 ===")
    print(cleaned_df.head())
    
    return merged_df, cleaned_df

if __name__ == "__main__":
    print("开始合并CAMP和grampa数据集...")
    merged_data, cleaned_data = merge_datasets()
    print("\n数据集合并和清洗完成！")
    print(f"最终正样本AMP数据集包含 {len(cleaned_data)} 个唯一序列")
