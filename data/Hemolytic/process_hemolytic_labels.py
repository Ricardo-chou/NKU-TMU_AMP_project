#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
溶血性标签二分类处理脚本

处理CAMP溶血性数据，将各种格式的溶血性标签转换为二分类标签：
- 高毒性 (1): 溶血性强
- 低毒性 (0): 溶血性弱或无溶血性

作者: AI Assistant
日期: 2025-01-15
"""

import pandas as pd
import re
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class HemolyticLabelProcessor:
    """溶血性标签处理器"""
    
    def __init__(self):
        # 半效应浓度类关键词
        self.concentration_keywords = [
            'LC50', 'HC50', 'MHC', 'HL50', 'IH50', 'EC50', 'LD50', 'HC10',
            'LC₅₀', 'HC₅₀', 'HL₅₀', 'IH₅₀', 'EC₅₀', 'LD₅₀'
        ]
        
        # 单位标准化映射
        self.unit_mapping = {
            'microm': 'microM',
            'μm': 'microM', 
            'µm': 'microM',
            'um': 'microM',
            'mircom': 'microM',  # 拼写错误
            'mircoM': 'microM',  # 拼写错误
            'microg/ml': 'microg/ml',
            'μg/ml': 'microg/ml',
            'µg/ml': 'microg/ml',
            'ug/ml': 'microg/ml',
            'mg/l': 'microg/ml',  # 1 mg/L = 1 μg/mL
            'mg/L': 'microg/ml'
        }
        
        # 非溶血性关键词
        self.non_hemolytic_keywords = [
            'non-hemolytic', 'nonhemolytic', 'no hemolytic', 
            'maximum nonhemolytic', 'non hemolytic',
            'Non-hemolytic', 'Nonhemolytic', 'No hemolytic',
            'Maximum nonhemolytic', 'Non hemolytic',
            'no hemolysis', 'no effect', 'not hemolytic',
            'No hemolysis', 'No effect', 'Not hemolytic'
        ]
        
        # 低毒性描述关键词
        self.low_toxicity_keywords = [
            'low hemolytic activity', 'weak hemolytic activity',
            'Low hemolytic activity', 'Weak hemolytic activity',
            'minimal hemolytic', 'slight hemolytic',
            'Minimal hemolytic', 'Slight hemolytic'
        ]
        
        # 高毒性描述关键词  
        self.high_toxicity_keywords = [
            'has hemolytic activity', 'exhibits hemolysis',
            'Has hemolytic activity', 'Exhibits hemolysis',
            'hemolytic activity', 'strong hemolytic',
            'Hemolytic activity', 'Strong hemolytic',
            'hemolytic against', 'hemolytic acitivity',
            'Hemolytic against', 'Hemolytic acitivity'
        ]
        
        # 简单细胞类型描述（通常表示缺乏具体数据）
        self.simple_cell_descriptions = [
            'human rbc', 'human erythrocytes', 'rat rbc', 'sheep rbcs',
            'Human RBC', 'Human erythrocytes', 'Rat RBC', 'Sheep RBCs',
            'duck erythrocytes', 'Duck erythrocytes'
        ]
        
        # 统计信息
        self.stats = {
            'total': 0,
            'concentration_type': 0,
            'percentage_type': 0,
            'non_hemolytic': 0,
            'multi_data': 0,
            'non_rbc': 0,
            'low_toxicity_desc': 0,
            'high_toxicity_desc': 0,
            'missing_data': 0,
            'simple_cell_desc': 0,
            'unparseable': 0,
            'high_toxic': 0,
            'low_toxic': 0
        }
    
    def normalize_unit(self, unit: str) -> str:
        """标准化单位"""
        unit = unit.strip().lower()
        return self.unit_mapping.get(unit, unit)
    
    def extract_concentration_value(self, text: str) -> Tuple[Optional[float], Optional[str], str]:
        """
        提取浓度值和单位
        返回: (数值, 单位, 比较符号)
        """
        # 处理科学计数法 (如: 2.3 x 10^5)
        sci_pattern = r'(\d+\.?\d*)\s*[x×]\s*10\^?([+-]?\d+)'
        sci_match = re.search(sci_pattern, text, re.IGNORECASE)
        if sci_match:
            base = float(sci_match.group(1))
            exp = int(sci_match.group(2))
            value = base * (10 ** exp)
            # 提取单位
            unit_pattern = r'(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
            unit_match = re.search(unit_pattern, text, re.IGNORECASE)
            unit = self.normalize_unit(unit_match.group(1)) if unit_match else 'microM'
            return value, unit, '='
        
        # 提取数值、比较符号和单位（包括特殊符号如>>、<<）
        pattern = r'([<>≤≥]{1,2})\s*(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            operator = match.group(1)
            value = float(match.group(2))
            unit = self.normalize_unit(match.group(3))
            return value, unit, operator
        
        # 如果没有比较符号，尝试直接匹配数值
        pattern2 = r'(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        match = re.search(pattern2, text, re.IGNORECASE)
        
        if match:
            value = float(match.group(1))
            unit = self.normalize_unit(match.group(2))
            return value, unit, '='
        
        return None, None, '='
    
    def extract_percentage_data(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
        """
        提取百分比溶血数据
        返回: (百分比, 浓度, 单位, 比较符号)
        """
        # 匹配标准百分比溶血模式
        # 例: 25% hemolysis at 10 microM, <5% hemolysis at 32 microM
        pattern1 = r'([<>≤≥]?)\s*(\d+\.?\d*)\s*[±]?\s*\d*\.?\d*\s*%\s*hemolysis\s*at\s*([<>≤≥]?)\s*(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        
        match = re.search(pattern1, text, re.IGNORECASE)
        if match:
            perc_operator = match.group(1) or '='
            percentage = float(match.group(2))
            conc_operator = match.group(3) or '='
            concentration = float(match.group(4))
            unit = self.normalize_unit(match.group(5))
            
            # 处理百分比的比较符号
            if perc_operator in ['<', '<=', '≤']:
                percentage = percentage  # 保持原值，表示最多这么多
            elif perc_operator in ['>', '>=', '≥']:
                percentage = percentage  # 保持原值，表示至少这么多
                
            # 处理浓度的比较符号  
            final_operator = conc_operator
            
            return percentage, concentration, unit, final_operator
        
        # 匹配逗号分隔格式: "100% hemolysis, 25 microg/ml"
        pattern2 = r'(\d+\.?\d*)\s*%\s*hemolysis\s*,\s*(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            percentage = float(match.group(1))
            concentration = float(match.group(2))
            unit = self.normalize_unit(match.group(3))
            
            return percentage, concentration, unit, '='
        
        # 匹配括号内百分比格式: "Human RBC(%haemolysis>10%at 8microM)"
        pattern4 = r'([<>≤≥]?)\s*(\d+\.?\d*)\s*%\s*(?:haemolysis|hemolysis)\s*(?:at|=)\s*([<>≤≥]?)\s*(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        
        match = re.search(pattern4, text, re.IGNORECASE)
        if match:
            perc_operator = match.group(1) or '='
            percentage = float(match.group(2))
            conc_operator = match.group(3) or '='
            concentration = float(match.group(4))
            unit = self.normalize_unit(match.group(5))
            
            return percentage, concentration, unit, conc_operator
        
        # 匹配方括号格式: "Human RBCs [10% hemolysis = 64 microM]"
        pattern5 = r'\[(\d+\.?\d*)\s*%\s*hemolysis\s*=\s*([<>≤≥]?)\s*(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)\]'
        
        match = re.search(pattern5, text, re.IGNORECASE)
        if match:
            percentage = float(match.group(1))
            conc_operator = match.group(2) or '='
            concentration = float(match.group(3))
            unit = self.normalize_unit(match.group(4))
            
            return percentage, concentration, unit, conc_operator
        
        # 匹配"hemolysis observed in the concentration/range of X"格式
        pattern3 = r'hemolysis\s+observed\s+in\s+the\s+(?:concentration|range)\s+of\s+(\d+\.?\d*)\s*(microm|μm|µm|um|mircom|mircoM|microg/ml|μg/ml|µg/ml|ug/ml|mg/l|mg/L)'
        
        match = re.search(pattern3, text, re.IGNORECASE)
        if match:
            concentration = float(match.group(1))
            unit = self.normalize_unit(match.group(2))
            # 假设观察到溶血即为有效，设为50%
            return 50.0, concentration, unit, '='
        
        return None, None, None, '='
    
    def is_non_hemolytic(self, text: str) -> bool:
        """检查是否为非溶血性描述"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.non_hemolytic_keywords)
    
    def is_low_toxicity_description(self, text: str) -> bool:
        """检查是否为低毒性描述"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.low_toxicity_keywords)
    
    def is_high_toxicity_description(self, text: str) -> bool:
        """检查是否为高毒性描述"""
        text_lower = text.lower()
        # 排除已经被其他规则处理的情况
        if self.is_non_hemolytic(text) or self.is_low_toxicity_description(text):
            return False
        return any(keyword in text_lower for keyword in self.high_toxicity_keywords)
    
    def is_missing_data(self, text: str) -> bool:
        """检查是否为缺失数据"""
        text = text.strip().lower()
        missing_patterns = [
            '- at na', 'at na', 'na', '', 'nan', 'null', 'none'
        ]
        return text in missing_patterns or 'at na' in text
    
    def is_simple_cell_description(self, text: str) -> bool:
        """检查是否为简单的细胞类型描述（缺乏具体数据）"""
        text = text.strip()
        return text in self.simple_cell_descriptions
    
    def is_non_rbc(self, text: str) -> bool:
        """检查是否为非红细胞实验"""
        non_rbc_keywords = [
            'hek293', 'nih 3t3', 'fibroblast', 'hela', 'cos', 
            'vero', 'mdck', 'cho', 'l929', 'a549'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in non_rbc_keywords)
    
    def extract_multiple_data(self, text: str) -> List[Dict]:
        """提取多组数据"""
        results = []
        
        # 分割多组数据 (用逗号或分号分割)
        parts = re.split(r'[,;]', text)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # 尝试提取浓度型数据
            for keyword in self.concentration_keywords:
                if keyword.lower() in part.lower():
                    value, unit, operator = self.extract_concentration_value(part)
                    if value is not None:
                        results.append({
                            'type': 'concentration',
                            'keyword': keyword,
                            'value': value,
                            'unit': unit,
                            'operator': operator
                        })
                        break
            
            # 尝试提取百分比型数据
            if 'hemolysis' in part.lower():
                percentage, concentration, unit, operator = self.extract_percentage_data(part)
                if percentage is not None and concentration is not None:
                    results.append({
                        'type': 'percentage',
                        'percentage': percentage,
                        'concentration': concentration,
                        'unit': unit,
                        'operator': operator
                    })
        
        return results
    
    def classify_concentration_type(self, value: float, unit: str, operator: str, keyword: str) -> int:
        """
        分类浓度型数据
        高毒 (1): ≤50 μM
        低毒 (0): >50 μM
        """
        if unit == 'microM':
            if operator in ['>', '>=', '≥']:
                # >50 microM 表示需要更高浓度才能达到效应，低毒
                return 0 if value > 50 else 1
            else:
                # ≤50 microM 表示低浓度就有效应，高毒
                return 1 if value <= 50 else 0
        elif unit == 'microg/ml':
            # 需要分子量信息，暂时标记为待处理
            return 0  # 保守处理，标记为低毒
        else:
            return 0  # 未知单位，保守处理
    
    def classify_percentage_type(self, percentage: float, concentration: float, unit: str, operator: str) -> int:
        """
        分类百分比型数据
        高毒 (1): 溶血率≥20% 且浓度≤64 μM
        低毒 (0): 其他情况
        """
        if unit == 'microM':
            # 处理浓度的比较符号
            if operator in ['<', '<=', '≤']:
                effective_conc = concentration
            elif operator in ['>', '>=', '≥']:
                effective_conc = concentration
            else:
                effective_conc = concentration
            
            # 判断规则：≥20% 且 ≤64 μM
            if percentage >= 20 and effective_conc <= 64:
                return 1
            else:
                return 0
        elif unit == 'microg/ml':
            # 需要分子量信息，保守处理
            return 0 if percentage < 20 else 1
        else:
            return 0
    
    def process_single_record(self, hemolytic_activity: str) -> Tuple[int, str, Dict]:
        """
        处理单条记录
        返回: (分类结果, 处理类型, 详细信息)
        """
        if pd.isna(hemolytic_activity) or hemolytic_activity.strip() == '':
            return 0, 'missing', {'reason': 'Missing data'}
        
        text = str(hemolytic_activity).strip()
        
        # 检查是否为缺失数据
        if self.is_missing_data(text):
            self.stats['missing_data'] += 1
            return 0, 'missing_data', {'reason': 'Missing or NA data', 'text': text}
        
        # 检查是否为非红细胞实验
        if self.is_non_rbc(text):
            self.stats['non_rbc'] += 1
            return 0, 'non_rbc', {'reason': 'Non-RBC experiment', 'text': text}
        
        # 检查是否为显式非溶血性
        if self.is_non_hemolytic(text):
            self.stats['non_hemolytic'] += 1
            return 0, 'non_hemolytic', {'reason': 'Explicitly non-hemolytic', 'text': text}
        
        # 检查是否为低毒性描述
        if self.is_low_toxicity_description(text):
            self.stats['low_toxicity_desc'] += 1
            return 0, 'low_toxicity_desc', {'reason': 'Low toxicity description', 'text': text}
        
        # 检查是否为高毒性描述
        if self.is_high_toxicity_description(text):
            self.stats['high_toxicity_desc'] += 1
            return 1, 'high_toxicity_desc', {'reason': 'High toxicity description', 'text': text}
        
        # 检查是否为简单细胞描述（缺乏具体数据）
        if self.is_simple_cell_description(text):
            self.stats['simple_cell_desc'] += 1
            return 0, 'simple_cell_desc', {'reason': 'Simple cell description without data', 'text': text}
        
        # 提取多组数据
        multi_data = self.extract_multiple_data(text)
        
        if len(multi_data) > 1:
            self.stats['multi_data'] += 1
            # 保守策略：取最严重的情况
            classifications = []
            for data in multi_data:
                if data['type'] == 'concentration':
                    cls = self.classify_concentration_type(
                        data['value'], data['unit'], data['operator'], data['keyword']
                    )
                    classifications.append(cls)
                elif data['type'] == 'percentage':
                    cls = self.classify_percentage_type(
                        data['percentage'], data['concentration'], data['unit'], data['operator']
                    )
                    classifications.append(cls)
            
            if classifications:
                final_cls = max(classifications)  # 取最高毒性
                return final_cls, 'multi_data', {'data': multi_data, 'classifications': classifications}
        
        # 单组数据处理
        if multi_data:
            data = multi_data[0]
            if data['type'] == 'concentration':
                self.stats['concentration_type'] += 1
                cls = self.classify_concentration_type(
                    data['value'], data['unit'], data['operator'], data['keyword']
                )
                return cls, 'concentration', data
            elif data['type'] == 'percentage':
                self.stats['percentage_type'] += 1
                cls = self.classify_percentage_type(
                    data['percentage'], data['concentration'], data['unit'], data['operator']
                )
                return cls, 'percentage', data
        
        # 尝试其他模式匹配
        # 检查是否包含浓度关键词但未能解析
        has_concentration_keyword = any(kw.lower() in text.lower() for kw in self.concentration_keywords)
        has_hemolysis = 'hemolysis' in text.lower()
        
        if has_concentration_keyword or has_hemolysis:
            self.stats['unparseable'] += 1
            return 0, 'unparseable', {'reason': 'Contains keywords but unparseable', 'text': text}
        
        # 默认情况
        self.stats['unparseable'] += 1
        return 0, 'unparseable', {'reason': 'No recognizable pattern', 'text': text}
    
    def process_dataframe(self, df: pd.DataFrame, activity_column: str = 'Hemolytic Activity') -> pd.DataFrame:
        """处理整个数据框"""
        print("开始处理溶血性标签...")
        
        # 重置统计信息
        self.stats = {k: 0 for k in self.stats.keys()}
        self.stats['total'] = len(df)
        
        results = []
        details = []
        process_types = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"处理进度: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
            
            classification, process_type, detail = self.process_single_record(row[activity_column])
            results.append(classification)
            process_types.append(process_type)
            details.append(detail)
            
            if classification == 1:
                self.stats['high_toxic'] += 1
            else:
                self.stats['low_toxic'] += 1
        
        # 添加结果到数据框
        df_result = df.copy()
        df_result['hemolytic_binary'] = results
        df_result['process_type'] = process_types
        df_result['process_detail'] = details
        
        print("\n处理完成！")
        self.print_statistics()
        
        return df_result
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n=== 处理统计信息 ===")
        print(f"总记录数: {self.stats['total']}")
        print(f"浓度型 (LC50/HC50等): {self.stats['concentration_type']} ({self.stats['concentration_type']/self.stats['total']*100:.1f}%)")
        print(f"百分比型: {self.stats['percentage_type']} ({self.stats['percentage_type']/self.stats['total']*100:.1f}%)")
        print(f"非溶血性: {self.stats['non_hemolytic']} ({self.stats['non_hemolytic']/self.stats['total']*100:.1f}%)")
        print(f"低毒性描述: {self.stats['low_toxicity_desc']} ({self.stats['low_toxicity_desc']/self.stats['total']*100:.1f}%)")
        print(f"高毒性描述: {self.stats['high_toxicity_desc']} ({self.stats['high_toxicity_desc']/self.stats['total']*100:.1f}%)")
        print(f"多组数据: {self.stats['multi_data']} ({self.stats['multi_data']/self.stats['total']*100:.1f}%)")
        print(f"非红细胞: {self.stats['non_rbc']} ({self.stats['non_rbc']/self.stats['total']*100:.1f}%)")
        print(f"缺失数据: {self.stats['missing_data']} ({self.stats['missing_data']/self.stats['total']*100:.1f}%)")
        print(f"简单细胞描述: {self.stats['simple_cell_desc']} ({self.stats['simple_cell_desc']/self.stats['total']*100:.1f}%)")
        print(f"无法解析: {self.stats['unparseable']} ({self.stats['unparseable']/self.stats['total']*100:.1f}%)")
        print("\n=== 分类结果 ===")
        print(f"高毒性 (1): {self.stats['high_toxic']} ({self.stats['high_toxic']/self.stats['total']*100:.1f}%)")
        print(f"低毒性 (0): {self.stats['low_toxic']} ({self.stats['low_toxic']/self.stats['total']*100:.1f}%)")

def main():
    """主函数"""
    # 读取数据
    input_file = "/Users/ricardozhao/PycharmProjects/AMP/data/Hemolytic/CAMP_hemolysis_data copy.csv"
    output_file = "/Users/ricardozhao/PycharmProjects/AMP/data/Hemolytic/CAMP_hemolysis_processed.csv"
    
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 创建处理器并处理数据
    processor = HemolyticLabelProcessor()
    df_processed = processor.process_dataframe(df)
    
    # 保存结果
    df_processed.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    # 显示一些示例
    print("\n=== 处理示例 ===")
    for process_type in ['concentration', 'percentage', 'non_hemolytic', 'low_toxicity_desc', 'high_toxicity_desc', 'missing_data', 'simple_cell_desc', 'unparseable']:
        examples = df_processed[df_processed['process_type'] == process_type].head(3)
        if not examples.empty:
            print(f"\n{process_type.upper()} 类型示例:")
            for idx, row in examples.iterrows():
                print(f"  原文: {row['Hemolytic Activity']}")
                print(f"  分类: {row['hemolytic_binary']} ({'高毒' if row['hemolytic_binary'] == 1 else '低毒'})")
                print(f"  详情: {row['process_detail']}")
                print()

if __name__ == "__main__":
    main()
