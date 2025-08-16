#!/usr/bin/env python3
"""
简化版AMP筛选器测试脚本
"""

import sys
import pandas as pd
import numpy as np
sys.path.append('/root/NKU-TMU_AMP_project')
from simple_amp_screener import SimpleAMPScreener

def test_small_batch():
    """小批量测试"""
    print("=== 简化版AMP筛选器小批量测试 ===")
    
    # 读取候选序列的前100个进行测试
    input_file = '/root/NKU-TMU_AMP_project/decode/filtered_candidate_sequences.csv'
    df = pd.read_csv(input_file)
    
    # 取前100个序列进行测试
    test_sequences = df['aa_seq'].head(100).tolist()
    print(f"测试序列数: {len(test_sequences)}")
    
    # 创建筛选器
    screener = SimpleAMPScreener(
        model_dir='/root/NKU-TMU_AMP_project/model_outputs',
        features_dir='/root/NKU-TMU_AMP_project/features'
    )
    
    try:
        # 加载模型
        screener.load_models()
        
        # 执行筛选
        screening_results = screener.screen_sequences(test_sequences, top_k=20)
        
        # 显示结果
        selection = screening_results['selection_results']
        print(f"\n=== 测试结果摘要 ===")
        print(f"输入序列: {len(test_sequences)}")
        print(f"选中序列: {len(selection['selected_sequences'])}")
        
        print(f"\n=== Top 10 选中序列 ===")
        for i in range(min(10, len(selection['selected_sequences']))):
            seq = selection['selected_sequences'][i]
            score = selection['selected_scores'][i]
            tier = selection['selected_tiers'][i]
            print(f"{i+1:2d}. [{tier}] {seq} (分数: {score:.4f})")
        
        # 保存测试结果
        output_prefix = '/root/NKU-TMU_AMP_project/decode/test_screening'
        screener.save_results(screening_results, output_prefix)
        
        print(f"\n✅ 小批量测试成功！结果保存在: {output_prefix}_*")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_by_step():
    """逐步测试各个组件"""
    print("\n=== 逐步组件测试 ===")
    
    # 读取少量序列
    input_file = '/root/NKU-TMU_AMP_project/decode/filtered_candidate_sequences.csv'
    df = pd.read_csv(input_file)
    test_sequences = df['aa_seq'].head(10).tolist()
    
    screener = SimpleAMPScreener(
        model_dir='/root/NKU-TMU_AMP_project/model_outputs',
        features_dir='/root/NKU-TMU_AMP_project/features'
    )
    
    screener.load_models()
    
    # 特征提取测试
    print("\n1. 特征提取测试...")
    features = screener.extract_features(test_sequences)
    print(f"   ✓ 序列特征形状: {features['seq_features'].shape}")
    print(f"   ✓ 条件特征形状: {features['cond_features'].shape}")
    
    # Step 1测试
    print("\n2. A模型快筛测试...")
    gate_results = screener.step1_gate_screening(features)
    print(f"   ✓ 门控通过: {gate_results['gate_pass'].sum()}/{len(test_sequences)}")
    
    # Step 2测试
    print("\n3. B₂双头精筛测试...")
    dual_results = screener.step2_dual_head_screening(features, gate_results)
    print(f"   ✓ 双菌株通过: {dual_results['dual_pass'].sum()}/{len(test_sequences)}")
    
    # Step 3测试
    print("\n4. B₁面板评估测试...")
    panel_results = screener.step3_panel_evaluation(features, gate_results, dual_results)
    print(f"   ✓ 平均命中率: {np.mean(panel_results['hit_at_10']):.3f}")
    
    # 最终分数测试
    print("\n5. 最终分数测试...")
    final_scores = screener.calculate_final_scores(gate_results, dual_results, panel_results)
    print(f"   ✓ 分数范围: [{np.min(final_scores):.3f}, {np.max(final_scores):.3f}]")
    
    print("\n✅ 所有组件测试通过！")

if __name__ == "__main__":
    print("开始简化版AMP筛选器测试...")
    
    # 小批量测试
    if test_small_batch():
        # 逐步测试
        test_step_by_step()
        print("\n🎉 所有测试完成！筛选器功能正常。")
    else:
        print("\n💥 测试失败，请检查模型文件和数据。")

