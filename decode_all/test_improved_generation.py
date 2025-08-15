#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test improved generation strategies to reduce E bias
"""

import os, sys
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from collections import Counter

def load_small_sample(pt_path: str, n_samples: int = 16):
    """加载少量样本用于调试"""
    data = torch.load(pt_path, map_location="cpu")
    embeds = data["embeddings"][:n_samples]
    masks = data["masks"][:n_samples]
    lengths = data.get("lengths", None)
    if lengths is not None:
        lengths = lengths[:n_samples]
    return embeds, masks, lengths

def decode_ids_to_sequence(token_ids, tokenizer):
    """解码token IDs到氨基酸序列"""
    sequence = []
    for token_id in token_ids:
        if token_id == tokenizer.pad_token_id:
            continue
        if token_id == tokenizer.eos_token_id:
            break
        
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        if token_str.startswith('▁') and len(token_str) == 2:
            aa_char = token_str[1:]
            if aa_char in "ACDEFGHIKLMNPQRSTVWY":
                sequence.append(aa_char)
    
    return "".join(sequence)

def test_generation_strategies(model, tokenizer, device, embeds, masks):
    """测试不同的生成策略"""
    
    strategies = [
        {
            "name": "baseline_temp0.8",
            "max_new_tokens": 32,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1.0
        },
        {
            "name": "higher_temp1.2", 
            "max_new_tokens": 32,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 1.2,
            "top_p": 0.9,
            "repetition_penalty": 1.0
        },
        {
            "name": "with_repetition_penalty",
            "max_new_tokens": 32,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.2  # 惩罚重复
        },
        {
            "name": "top_k_sampling",
            "max_new_tokens": 32,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 10,  # 限制候选token数量
            "top_p": 1.0,
            "repetition_penalty": 1.1
        },
        {
            "name": "diverse_beam",
            "max_new_tokens": 32,
            "num_beams": 3,
            "do_sample": False,
            "num_beam_groups": 3,
            "diversity_penalty": 0.5,
            "repetition_penalty": 1.1
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n=== 测试策略: {strategy['name']} ===")
        
        # 设置decoder start token
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
        
        # 构建encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=embeds.to(device))
        
        gen_kwargs = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": masks.to(device),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            **{k: v for k, v in strategy.items() if k != "name"}
        }
        
        try:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
            
            # 解码所有序列
            sequences = []
            for i in range(output_ids.shape[0]):
                seq = decode_ids_to_sequence(output_ids[i].tolist(), tokenizer)
                sequences.append(seq)
            
            # 分析氨基酸组成
            all_aas = ''.join(sequences)
            aa_counts = Counter(all_aas)
            total_aa = len(all_aas)
            
            if total_aa > 0:
                e_ratio = aa_counts.get('E', 0) / total_aa * 100
                l_ratio = aa_counts.get('L', 0) / total_aa * 100
                
                print(f"生成序列数: {len(sequences)}")
                print(f"平均长度: {total_aa/len(sequences):.1f}")
                print(f"E含量: {e_ratio:.1f}%")
                print(f"L含量: {l_ratio:.1f}%")
                
                # 显示前3个序列
                print("示例序列:")
                for i, seq in enumerate(sequences[:3]):
                    print(f"  {i+1}: {seq}")
                
                results[strategy['name']] = {
                    'sequences': sequences,
                    'e_ratio': e_ratio,
                    'l_ratio': l_ratio,
                    'avg_length': total_aa/len(sequences),
                    'aa_counts': aa_counts
                }
            else:
                print("未生成有效序列")
                results[strategy['name']] = None
                
        except Exception as e:
            print(f"策略失败: {e}")
            results[strategy['name']] = None
    
    return results

def analyze_results(results):
    """分析不同策略的结果"""
    print(f"\n=== 策略对比分析 ===")
    print("策略\t\tE含量\tL含量\t平均长度")
    print("-" * 50)
    
    best_strategy = None
    best_e_ratio = float('inf')
    
    for name, result in results.items():
        if result is not None:
            e_ratio = result['e_ratio']
            l_ratio = result['l_ratio']
            avg_len = result['avg_length']
            
            print(f"{name:<20}\t{e_ratio:.1f}%\t{l_ratio:.1f}%\t{avg_len:.1f}")
            
            # 找到E含量最低的策略
            if e_ratio < best_e_ratio:
                best_e_ratio = e_ratio
                best_strategy = name
        else:
            print(f"{name:<20}\t失败\t失败\t失败")
    
    if best_strategy:
        print(f"\n🏆 最佳策略: {best_strategy} (E含量: {best_e_ratio:.1f}%)")
        
        # 显示最佳策略的详细氨基酸分布
        best_result = results[best_strategy]
        aa_counts = best_result['aa_counts']
        total_aa = sum(aa_counts.values())
        
        print(f"\n{best_strategy} 的氨基酸分布:")
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            count = aa_counts.get(aa, 0)
            if count > 0:
                ratio = count / total_aa * 100
                print(f"  {aa}: {ratio:.1f}%")
    
    return best_strategy

def main():
    pt_path = "/root/NKU-TMU_AMP_project/generated_embeddings.pt"
    model_dir = "/root/autodl-tmp/prot_t5_xl_uniref50"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载数据
    embeds, masks, lengths = load_small_sample(pt_path, n_samples=16)
    print(f"加载了 {embeds.shape[0]} 个样本")
    
    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    print("模型加载完成")
    
    # 测试不同策略
    results = test_generation_strategies(model, tokenizer, device, embeds, masks)
    
    # 分析结果
    best_strategy = analyze_results(results)
    
    return best_strategy, results

if __name__ == "__main__":
    best_strategy, results = main()
