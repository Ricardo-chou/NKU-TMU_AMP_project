#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
深度分析E含量异常高的原因
"""

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import matplotlib.pyplot as plt
from collections import Counter

def analyze_model_bias(model, tokenizer, device):
    """分析模型本身的偏向性"""
    print("=== 模型偏向性分析 ===")
    
    # 1. 检查LM head权重
    lm_head = model.lm_head
    lm_weights = lm_head.weight.data  # [vocab_size, hidden_size]
    
    print(f"LM head权重形状: {lm_weights.shape}")
    
    # 分析氨基酸token的权重分布
    aa_tokens = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        token_id = tokenizer.convert_tokens_to_ids(f"▁{aa}")
        if token_id != tokenizer.unk_token_id:
            aa_tokens[aa] = token_id
    
    print(f"找到 {len(aa_tokens)} 个氨基酸token")
    
    # 计算每个氨基酸token权重的统计信息
    aa_weight_stats = {}
    for aa, token_id in aa_tokens.items():
        if token_id < lm_weights.shape[0]:
            weight_vec = lm_weights[token_id]
            stats = {
                'mean': weight_vec.mean().item(),
                'std': weight_vec.std().item(),
                'norm': torch.norm(weight_vec).item(),
                'max': weight_vec.max().item(),
                'min': weight_vec.min().item()
            }
            aa_weight_stats[aa] = stats
    
    # 按权重范数排序
    sorted_by_norm = sorted(aa_weight_stats.items(), key=lambda x: x[1]['norm'], reverse=True)
    
    print("\n氨基酸token权重范数排序:")
    print("AA\tToken_ID\tNorm\tMean\tStd")
    print("-" * 50)
    for aa, stats in sorted_by_norm:
        token_id = aa_tokens[aa]
        print(f"{aa}\t{token_id}\t{stats['norm']:.3f}\t{stats['mean']:.3f}\t{stats['std']:.3f}")
    
    # 检查E token的特殊性
    e_token_id = aa_tokens.get('E')
    if e_token_id:
        e_stats = aa_weight_stats['E']
        print(f"\n*** E token (id={e_token_id}) 分析 ***")
        print(f"权重范数: {e_stats['norm']:.6f}")
        print(f"权重均值: {e_stats['mean']:.6f}")
        print(f"权重标准差: {e_stats['std']:.6f}")
        
        # 与其他氨基酸比较
        avg_norm = np.mean([stats['norm'] for stats in aa_weight_stats.values()])
        print(f"平均权重范数: {avg_norm:.6f}")
        print(f"E相对偏差: {(e_stats['norm'] - avg_norm) / avg_norm * 100:.1f}%")
    
    return aa_tokens, aa_weight_stats

def analyze_embeddings_bias(embeddings_path, n_samples=100):
    """分析输入embeddings的偏向性"""
    print("\n=== 输入Embeddings偏向性分析 ===")
    
    data = torch.load(embeddings_path, map_location="cpu")
    embeds = data["embeddings"][:n_samples]  # [N, 48, 1024]
    masks = data["masks"][:n_samples]       # [N, 48]
    
    print(f"分析 {n_samples} 个样本的embeddings")
    
    # 1. 整体统计
    valid_embeds = embeds[masks]  # 只考虑有效位置
    print(f"有效embedding数量: {valid_embeds.shape[0]}")
    print(f"Embedding统计:")
    print(f"  均值: {valid_embeds.mean().item():.6f}")
    print(f"  标准差: {valid_embeds.std().item():.6f}")
    print(f"  最小值: {valid_embeds.min().item():.6f}")
    print(f"  最大值: {valid_embeds.max().item():.6f}")
    
    # 2. 检查embedding的分布特征
    # 计算每个位置的embedding向量的特征
    position_stats = []
    for i in range(embeds.shape[1]):  # 对每个位置
        pos_embeds = embeds[:, i, :][masks[:, i]]  # 该位置的所有有效embedding
        if len(pos_embeds) > 0:
            stats = {
                'position': i,
                'count': len(pos_embeds),
                'mean_norm': torch.norm(pos_embeds, dim=1).mean().item(),
                'mean_value': pos_embeds.mean().item(),
                'std_value': pos_embeds.std().item()
            }
            position_stats.append(stats)
    
    # 检查是否有位置偏向
    if position_stats:
        norms = [s['mean_norm'] for s in position_stats]
        print(f"\n位置embedding范数统计:")
        print(f"  最小范数: {min(norms):.6f}")
        print(f"  最大范数: {max(norms):.6f}")
        print(f"  范数标准差: {np.std(norms):.6f}")
    
    return position_stats

def test_generation_without_embeddings(model, tokenizer, device):
    """测试不使用ProT-Diff embeddings的生成"""
    print("\n=== 测试纯随机生成 ===")
    
    # 创建随机encoder outputs
    batch_size = 8
    seq_len = 32
    hidden_size = 1024
    
    # 随机初始化encoder outputs
    random_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.1
    random_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    encoder_outputs = BaseModelOutput(last_hidden_state=random_embeds)
    
    # 确保decoder配置
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    gen_kwargs = {
        "encoder_outputs": encoder_outputs,
        "attention_mask": random_mask,
        "max_new_tokens": 20,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
    
    # 解码并分析
    sequences = []
    for i in range(output_ids.shape[0]):
        seq = ""
        for token_id in output_ids[i].tolist():
            if token_id == tokenizer.pad_token_id:
                continue
            if token_id == tokenizer.eos_token_id:
                break
            
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if token_str.startswith('▁') and len(token_str) == 2:
                aa_char = token_str[1:]
                if aa_char in "ACDEFGHIKLMNPQRSTVWY":
                    seq += aa_char
        sequences.append(seq)
    
    # 分析氨基酸组成
    all_aas = ''.join(sequences)
    if all_aas:
        aa_counts = Counter(all_aas)
        total_aa = len(all_aas)
        e_ratio = aa_counts.get('E', 0) / total_aa * 100
        
        print(f"随机embedding生成的序列:")
        for i, seq in enumerate(sequences[:3]):
            print(f"  {i+1}: {seq}")
        
        print(f"\n随机embedding的E含量: {e_ratio:.1f}%")
        print("氨基酸分布:")
        for aa in sorted(aa_counts.keys()):
            ratio = aa_counts[aa] / total_aa * 100
            print(f"  {aa}: {ratio:.1f}%")
    else:
        print("随机embedding未生成有效序列")

def analyze_training_data_hypothesis():
    """分析训练数据偏向的假设"""
    print("\n=== 训练数据偏向假设分析 ===")
    
    print("可能的原因:")
    print("1. ProtT5训练时使用的蛋白质数据库可能有E含量偏向")
    print("2. ProT-Diff训练数据中抗菌肽可能富含带电氨基酸(E,D,K,R)")
    print("3. 模型学习到了特定的序列模式偏好")
    
    print("\n抗菌肽的典型特征:")
    print("- 富含带正电荷氨基酸 (K, R)")
    print("- 富含疏水性氨基酸 (L, V, I, F)")
    print("- 可能含有较多极性氨基酸用于膜相互作用")
    print("- E作为带负电荷氨基酸，可能在某些AMP中起重要作用")

def main():
    model_dir = "/root/autodl-tmp/prot_t5_xl_uniref50"
    embeddings_path = "/root/NKU-TMU_AMP_project/generated_embeddings.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载模型
    tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    # 1. 分析模型偏向性
    aa_tokens, aa_weight_stats = analyze_model_bias(model, tokenizer, device)
    
    # 2. 分析输入embeddings
    position_stats = analyze_embeddings_bias(embeddings_path, n_samples=100)
    
    # 3. 测试随机生成
    test_generation_without_embeddings(model, tokenizer, device)
    
    # 4. 分析训练数据假设
    analyze_training_data_hypothesis()
    
    print("\n=== 结论与建议 ===")
    print("E含量异常高的可能原因排序:")
    print("1. 🔥 模型权重偏向: LM head中E token权重可能异常")
    print("2. 🔥 ProT-Diff embeddings特征: 输入embeddings可能编码了E偏向")
    print("3. 🔥 训练数据偏向: 抗菌肽数据集可能天然富含E")
    print("4. 解码策略: 虽然已优化，但仍可能不够")
    
    print("\n进一步改进建议:")
    print("1. 尝试更强的repetition_penalty (1.5-2.0)")
    print("2. 使用更高的temperature (1.5-2.0)")
    print("3. 考虑后处理过滤异常高E含量的序列")
    print("4. 分析原始ProT-Diff训练数据的氨基酸分布")

if __name__ == "__main__":
    main()
