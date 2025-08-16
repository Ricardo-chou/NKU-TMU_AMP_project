#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 generated_embeddings.pt 文件的数据结构
"""

import torch
import numpy as np

def analyze_embeddings(file_path):
    """分析嵌入文件的数据结构"""
    print(f"正在加载文件: {file_path}")
    
    try:
        # 加载 .pt 文件
        data = torch.load(file_path, map_location='cpu')
        
        print("\n=== 文件基本信息 ===")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print("字典键:", list(data.keys()))
            for key, value in data.items():
                print(f"\n键 '{key}':")
                print(f"  类型: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  形状: {value.shape}")
                if hasattr(value, 'dtype'):
                    print(f"  数据类型: {value.dtype}")
                if isinstance(value, (list, tuple)):
                    print(f"  长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0])}")
                        if hasattr(value[0], 'shape'):
                            print(f"  第一个元素形状: {value[0].shape}")
        
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            print(f"张量/数组形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"设备: {data.device if hasattr(data, 'device') else 'N/A'}")
            
        elif isinstance(data, (list, tuple)):
            print(f"序列长度: {len(data)}")
            if len(data) > 0:
                print(f"第一个元素类型: {type(data[0])}")
                if hasattr(data[0], 'shape'):
                    print(f"第一个元素形状: {data[0].shape}")
                    
        # 如果是张量，显示一些统计信息
        if isinstance(data, torch.Tensor):
            print(f"\n=== 张量统计信息 ===")
            print(f"最小值: {data.min():.6f}")
            print(f"最大值: {data.max():.6f}")
            print(f"均值: {data.mean():.6f}")
            print(f"标准差: {data.std():.6f}")
            
        elif isinstance(data, dict) and any(isinstance(v, torch.Tensor) for v in data.values()):
            print(f"\n=== 张量统计信息 ===")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"键 '{key}':")
                    print(f"  最小值: {value.min():.6f}")
                    print(f"  最大值: {value.max():.6f}")
                    print(f"  均值: {value.mean():.6f}")
                    print(f"  标准差: {value.std():.6f}")
        
        return data
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

if __name__ == "__main__":
    file_path = "/root/NKU-TMU_AMP_project/generated_embeddings.pt"
    data = analyze_embeddings(file_path)

