#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ProT-Diff Embeddings解码与分析一体化脚本

功能：
1. 解码ProT-Diff生成的embeddings为氨基酸序列
2. 分析氨基酸组成和质量评估
3. 生成详细报告和可视化结果
4. 提供优化建议

作者：基于ProT-Diff和ProtT5的抗菌肽序列解码
"""

import os, sys, json, math, time, argparse, traceback
from typing import List, Optional, Dict, Tuple
from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# HF Transformers
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

def parse_args():
    p = argparse.ArgumentParser(description="解码并分析ProT-Diff embeddings")
    
    # 输入输出参数
    p.add_argument("--pt_path", type=str, required=True,
                   help="ProT-Diff生成的embeddings文件路径")
    p.add_argument("--model_dir", type=str, default="/root/autodl-tmp/prot_t5_xl_uniref50",
                   help="ProtT5模型目录")
    p.add_argument("--out_prefix", type=str, default="decoded",
                   help="输出文件前缀")
    
    # 解码参数
    p.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"],
                   help="运行设备")
    p.add_argument("--fp16", action="store_true", help="启用混合精度")
    p.add_argument("--max_new_tokens", type=int, default=48, help="最大生成长度")
    
    # 生成策略参数（推荐设置）
    p.add_argument("--num_beams", type=int, default=1, help="束搜索大小")
    p.add_argument("--temperature", type=float, default=1.2, help="采样温度（推荐1.2）")
    p.add_argument("--top_p", type=float, default=0.9, help="核采样参数（推荐0.9）")
    p.add_argument("--top_k", type=int, default=0, help="Top-k采样")
    p.add_argument("--repetition_penalty", type=float, default=1.5, help="重复惩罚（推荐1.5）")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3, help="防重复n-gram大小")
    
    # 后处理参数
    p.add_argument("--truncate_by_mask", action="store_true", default=True,
                   help="根据mask截断序列")
    p.add_argument("--filter_near_zero", action="store_true", default=True,
                   help="过滤近零embeddings")
    p.add_argument("--near_zero_threshold", type=float, default=1e-6,
                   help="近零过滤阈值")
    
    # 质量控制参数
    p.add_argument("--max_e_ratio", type=float, default=20.0,
                   help="最大E含量百分比（超过则标记为异常）")
    p.add_argument("--min_length", type=int, default=5,
                   help="最小序列长度")
    p.add_argument("--max_length", type=int, default=50,
                   help="最大序列长度")
    
    # 分析参数
    p.add_argument("--n_samples", type=int, default=None,
                   help="限制处理样本数（用于测试）")
    p.add_argument("--verbose", action="store_true", help="详细输出")
    p.add_argument("--generate_report", action="store_true", default=True,
                   help="生成分析报告")
    
    return p.parse_args()

def smart_device(arg: str) -> torch.device:
    """智能设备选择"""
    if arg == "cpu": return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pt(pt_path: str):
    """加载embeddings文件"""
    obj = torch.load(pt_path, map_location="cpu")
    for k in ["embeddings", "masks"]:
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {pt_path}")
    return obj

def validate_inputs(embeds: torch.Tensor, masks: torch.Tensor, lengths=None):
    """验证输入数据"""
    if embeds.ndim != 3:
        raise ValueError(f"Expected embeddings to be 3D, got {embeds.ndim}D")
    if embeds.shape[1] != 48:
        raise ValueError(f"Expected sequence length 48, got {embeds.shape[1]}")
    if embeds.shape[2] != 1024:
        raise ValueError(f"Expected embedding dim 1024, got {embeds.shape[2]}")
    if masks.shape[:2] != embeds.shape[:2]:
        raise ValueError(f"Mask shape {masks.shape} doesn't match embeddings {embeds.shape}")
    
    # 数据类型转换
    if embeds.dtype != torch.float32:
        print(f"[WARN] Converting embeddings from {embeds.dtype} to float32")
        embeds = embeds.float()
    if masks.dtype != torch.bool:
        print(f"[WARN] Converting masks from {masks.dtype} to bool")
        masks = masks.bool()
    
    # 数据质量检查
    if torch.isnan(embeds).any():
        print("[WARN] Found NaN values in embeddings")
    if torch.isinf(embeds).any():
        print("[WARN] Found Inf values in embeddings")
    
    valid_mask_counts = masks.sum(dim=1)
    if (valid_mask_counts == 0).any():
        print("[WARN] Found sequences with no valid positions in mask")
    
    return embeds, masks

def filter_near_zero_embeddings(embeds: torch.Tensor, masks: torch.Tensor, 
                                threshold: float = 1e-6, verbose: bool = False) -> torch.Tensor:
    """过滤近零embeddings提升稳定性"""
    embed_norms = torch.norm(embeds, dim=-1)  # [B, L]
    near_zero = embed_norms < threshold
    updated_masks = masks & (~near_zero)
    
    # 统计信息
    original_valid = masks.sum().item()
    filtered_valid = updated_masks.sum().item()
    filtered_count = original_valid - filtered_valid
    
    if verbose and filtered_count > 0:
        print(f"[DEBUG] Filtered {filtered_count}/{original_valid} near-zero positions")
    
    # 确保每个序列至少有一个有效位置
    valid_counts = updated_masks.sum(dim=1)
    restored_count = 0
    for i, count in enumerate(valid_counts):
        if count == 0:
            first_valid = masks[i].nonzero(as_tuple=True)[0]
            if len(first_valid) > 0:
                updated_masks[i, first_valid[0]] = True
                restored_count += 1
    
    if verbose and restored_count > 0:
        print(f"[DEBUG] Restored {restored_count} sequences that were completely filtered")
    
    return updated_masks

def decode_ids_to_sequence(token_ids: List[int], tokenizer) -> str:
    """将token IDs解码为氨基酸序列"""
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

def build_encoder_outputs(batch_embeds: torch.Tensor, attention_mask: torch.Tensor):
    """构建encoder outputs"""
    return BaseModelOutput(last_hidden_state=batch_embeds)

def decode_batch(model, tokenizer, device, embeds: torch.Tensor, masks: torch.Tensor,
                args, enable_near_zero_filter: bool = True) -> List[str]:
    """批量解码"""
    # 近零过滤
    if enable_near_zero_filter:
        threshold = getattr(args, 'near_zero_threshold', 1e-6)
        verbose = getattr(args, 'verbose', False)
        masks = filter_near_zero_embeddings(embeds, masks, threshold, verbose)
    
    # 采样策略
    do_sample = args.temperature != 1.0 or args.top_p < 1.0 or args.top_k > 0
    
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else 1.0,
        top_p=args.top_p if do_sample else 1.0,
        top_k=args.top_k if args.top_k > 0 else None,
        repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1.0 else None,
        no_repeat_ngram_size=args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
        attention_mask=masks,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        length_penalty=1.0,
    )
    
    # 确保decoder配置
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    # 构建encoder outputs
    enc_out = build_encoder_outputs(embeds, masks)
    
    try:
        # 生成
        if device.type == "cuda" and args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_ids = model.generate(encoder_outputs=enc_out, **gen_kwargs)
        else:
            out_ids = model.generate(encoder_outputs=enc_out, **gen_kwargs)
            
        # 解码
        sequences = []
        for i in range(out_ids.shape[0]):
            seq = decode_ids_to_sequence(out_ids[i].tolist(), tokenizer)
            sequences.append(seq)
        
        return sequences
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise

def trim_by_mask(seq: str, mask_row: torch.Tensor, length_meta: Optional[int]) -> str:
    """根据mask和长度信息截断序列"""
    if length_meta is not None and length_meta > 0 and length_meta <= len(seq):
        return seq[:length_meta]
    valid_len = int(mask_row.sum().item())
    if 0 < valid_len <= len(seq):
        return seq[:valid_len]
    return seq

def analyze_aa_composition(sequences: List[str]) -> Dict:
    """分析氨基酸组成"""
    if not sequences:
        return {}
    
    # 合并所有序列
    all_sequences = ''.join(seq for seq in sequences if seq)
    total_aa = len(all_sequences)
    
    if total_aa == 0:
        return {}
    
    # 统计氨基酸
    aa_counts = Counter(all_sequences)
    
    # 天然蛋白质氨基酸分布（参考值）
    natural_composition = {
        'A': 8.25, 'R': 5.53, 'N': 4.06, 'D': 5.45, 'C': 1.37,
        'Q': 3.93, 'E': 6.75, 'G': 7.07, 'H': 2.27, 'I': 5.96,
        'L': 9.66, 'K': 5.84, 'M': 2.42, 'F': 3.86, 'P': 4.70,
        'S': 6.56, 'T': 5.34, 'W': 1.08, 'Y': 2.92, 'V': 6.87
    }
    
    # 计算分析结果
    analysis = {
        'total_sequences': len(sequences),
        'valid_sequences': sum(1 for seq in sequences if len(seq) > 0),
        'total_aa': total_aa,
        'avg_length': total_aa / len(sequences) if sequences else 0,
        'aa_counts': aa_counts,
        'aa_frequencies': {aa: count/total_aa*100 for aa, count in aa_counts.items()},
        'natural_composition': natural_composition,
        'deviations': {},
        'fold_changes': {},
        'quality_flags': []
    }
    
    # 计算偏差和倍数变化
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        observed = analysis['aa_frequencies'].get(aa, 0)
        natural = natural_composition.get(aa, 0)
        
        deviation = observed - natural
        fold_change = observed / natural if natural > 0 else float('inf')
        
        analysis['deviations'][aa] = deviation
        analysis['fold_changes'][aa] = fold_change
    
    # 质量评估
    e_ratio = analysis['aa_frequencies'].get('E', 0)
    if e_ratio > 20:
        analysis['quality_flags'].append(f"E含量异常高: {e_ratio:.1f}%")
    elif e_ratio > 15:
        analysis['quality_flags'].append(f"E含量偏高: {e_ratio:.1f}%")
    
    # 检查其他异常
    for aa, fold in analysis['fold_changes'].items():
        if fold > 3:
            analysis['quality_flags'].append(f"{aa}含量异常高: {fold:.1f}倍")
        elif fold < 0.2:
            analysis['quality_flags'].append(f"{aa}含量异常低: {fold:.1f}倍")
    
    return analysis

def generate_quality_report(analysis: Dict, args) -> str:
    """生成质量评估报告"""
    if not analysis:
        return "无法生成报告：分析数据为空"
    
    report = []
    report.append("=" * 60)
    report.append("ProT-Diff序列质量分析报告")
    report.append("=" * 60)
    
    # 基本统计
    report.append(f"基本统计:")
    report.append(f"  总序列数: {analysis['total_sequences']}")
    report.append(f"  有效序列数: {analysis['valid_sequences']}")
    report.append(f"  有效率: {analysis['valid_sequences']/analysis['total_sequences']*100:.1f}%")
    report.append(f"  平均长度: {analysis['avg_length']:.1f}")
    report.append(f"  总氨基酸数: {analysis['total_aa']}")
    
    # 生成参数
    report.append(f"生成参数:")
    report.append(f"  温度: {args.temperature}")
    report.append(f"  重复惩罚: {args.repetition_penalty}")
    report.append(f"  Top-p: {args.top_p}")
    report.append(f"  N-gram防重复: {args.no_repeat_ngram_size}")
    
    # 氨基酸组成分析
    report.append(f"氨基酸组成分析:")
    report.append(f"{'AA':<3} {'观察%':<8} {'天然%':<8} {'偏差':<8} {'倍数':<8} {'状态'}")
    report.append("-" * 50)
    
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        observed = analysis['aa_frequencies'].get(aa, 0)
        natural = analysis['natural_composition'].get(aa, 0)
        deviation = analysis['deviations'].get(aa, 0)
        fold = analysis['fold_changes'].get(aa, 0)
        
        status = ""
        if fold > 2.5:
            status = "异常高"
        elif fold > 1.5:
            status = "偏高"
        elif fold < 0.3:
            status = "偏低"
        elif abs(deviation) < 1:
            status = "正常"
        
        report.append(f"{aa:<3} {observed:<8.1f} {natural:<8.1f} {deviation:<+8.1f} {fold:<8.1f} {status}")
    
    # 质量评估
    report.append(f"质量评估:")
    if analysis['quality_flags']:
        for flag in analysis['quality_flags']:
            report.append(f"{flag}")
    else:
        report.append(f"未发现明显异常")
    
    # E含量特别分析
    e_ratio = analysis['aa_frequencies'].get('E', 0)
    e_fold = analysis['fold_changes'].get('E', 0)
    report.append(f"E含量深度分析:")
    report.append(f"  观察值: {e_ratio:.1f}%")
    report.append(f"  天然值: {analysis['natural_composition']['E']:.1f}%")
    report.append(f"  倍数: {e_fold:.1f}x")
    
    if e_ratio > 20:
        report.append(f"  评估: 严重偏高，可能影响生物活性")
    elif e_ratio > 15:
        report.append(f"  评估: 明显偏高，建议进一步优化")
    elif e_ratio > 10:
        report.append(f"  评估: 轻度偏高，在可接受范围内")
    else:
        report.append(f"  评估: 正常范围")
    
    # 改进建议
    report.append(f"改进建议:")
    if e_ratio > 15:
        report.append(f"  1. 提高repetition_penalty至{args.repetition_penalty + 0.3:.1f}")
        report.append(f"  2. 提高temperature至{args.temperature + 0.3:.1f}")
        report.append(f"  3. 增加no_repeat_ngram_size至{args.no_repeat_ngram_size + 1}")
        report.append(f"  4. 考虑后处理过滤E含量>15%的序列")
    else:
        report.append(f"  当前参数设置合理，序列质量良好")
    
    report.append(f"结论:")
    if analysis['quality_flags']:
        report.append(f"  生成的序列存在一定程度的氨基酸分布偏向，")
        report.append(f"  主要体现在E含量偏高。这可能源于ProT-Diff训练数据")
        report.append(f"  的分布特征。建议结合生物学验证使用这些序列。")
    else:
        report.append(f"  生成的序列氨基酸分布相对均衡，质量良好。")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def save_results(out_rows: List[Dict], fasta_lines: List[str], analysis: Dict, 
                report: str, args):
    """保存结果文件"""
    # CSV文件
    csv_path = f"{args.out_prefix}.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","length_pred","length_meta","aa_seq","e_ratio","quality"])
        for r in out_rows:
            w.writerow([r["id"], r["length_pred"], r["length_meta"], 
                       r["aa_seq"], r.get("e_ratio", 0), r.get("quality", "unknown")])
    
    # FASTA文件
    fasta_path = f"{args.out_prefix}.fasta"
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.writelines(fasta_lines)
    
    # JSONL文件
    jsonl_path = f"{args.out_prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    
    # 分析报告
    if args.generate_report:
        report_path = f"{args.out_prefix}_analysis_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        # 分析数据JSON
        analysis_path = f"{args.out_prefix}_analysis.json"
        # 转换Counter对象为dict以便JSON序列化
        analysis_copy = analysis.copy()
        if 'aa_counts' in analysis_copy:
            analysis_copy['aa_counts'] = dict(analysis_copy['aa_counts'])
        
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_copy, f, ensure_ascii=False, indent=2)
    
    return csv_path, fasta_path, jsonl_path

def main():
    args = parse_args()
    # 在 main() 里 parse_args() 之后加
    if args.n_samples is not None:
        os.makedirs("test_data", exist_ok=True)
        args.out_prefix = os.path.join("test_data", args.out_prefix)
    else:
        os.makedirs("full_data", exist_ok=True)
        args.out_prefix = os.path.join("full_data", args.out_prefix)

    t0 = time.time()
    device = smart_device(args.device)
    
    print(f"ProT-Diff Embeddings解码与分析")
    print(f"使用设备: {device}")
    print(f"生成参数: temp={args.temperature}, rep_penalty={args.repetition_penalty}, top_p={args.top_p}")
    
    try:
        # 1. 加载数据
        print(f"加载数据...")
        pack = load_pt(args.pt_path)
        embeds = pack["embeddings"]
        masks = pack["masks"]
        lengths = pack.get("lengths", None)
        
        # 限制样本数
        if args.n_samples is not None:
            embeds = embeds[:args.n_samples]
            masks = masks[:args.n_samples]
            if lengths is not None:
                lengths = lengths[:args.n_samples]
            print(f"限制处理 {args.n_samples} 个样本")
        
        # 验证数据
        embeds, masks = validate_inputs(embeds, masks, lengths)
        N = embeds.shape[0]
        print(f"加载了 {N} 个样本，embedding形状: {embeds.shape},文件名为{args.pt_path},导出地址为{args.out_prefix}")
        
        # 2. 加载模型
        print(f"加载ProtT5模型...")
        if not os.path.exists(args.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {args.model_dir}")
        
        # 加载tokenizer
        try:
            tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=False)
        except Exception as e1:
            try:
                tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=True)
            except Exception as e2:
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir, legacy=False)
        
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        model = model.to(device)
        model.eval()
        
        print(f"模型加载完成")
        print(f"词汇表大小: {tokenizer.vocab_size}")
        
        # 内存优化
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"GPU内存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    
    # 3. 批量解码
    print(f"开始解码...")
    bs = args.batch_size
    out_rows = []
    fasta_lines = []
    
    # 处理lengths
    lengths_list = None
    if lengths is not None and torch.is_tensor(lengths):
        lengths_list = lengths.tolist()
    
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s+bs, N)
            batch_emb = embeds[s:e].to(device, non_blocking=True)
            batch_msk = masks[s:e].to(device, non_blocking=True)
            
            try:
                seqs = decode_batch(model, tokenizer, device, batch_emb, batch_msk, 
                                  args, enable_near_zero_filter=args.filter_near_zero)
            except Exception as ex:
                print(f"批次 {s}:{e} 解码失败: {ex}")
                print(f"尝试保守设置...")
                
                # 保守fallback设置
                class FallbackArgs:
                    def __init__(self, original_args):
                        self.max_new_tokens = original_args.max_new_tokens
                        self.num_beams = 1
                        self.temperature = 1.0
                        self.top_p = 0.95
                        self.top_k = 0
                        self.repetition_penalty = 1.2
                        self.no_repeat_ngram_size = 2
                        self.fp16 = False
                        self.filter_near_zero = False
                        self.verbose = getattr(original_args, 'verbose', False)
                        self.near_zero_threshold = original_args.near_zero_threshold
                
                try:
                    fallback_args = FallbackArgs(args)
                    seqs = decode_batch(model, tokenizer, device, batch_emb, batch_msk, 
                                      fallback_args, enable_near_zero_filter=False)
                    print(f"保守设置解码成功")
                except Exception as ex2:
                    print(f"保守设置也失败: {ex2}")
                    batch_size = batch_emb.shape[0]
                    seqs = [""] * batch_size
                    print(f"使用空序列占位")
            
            # 处理每个序列
            for i, seq in enumerate(seqs):
                idx = s + i
                seq_raw = seq
                length_meta = lengths_list[idx] if lengths_list is not None else None
                
                # 根据mask截断
                if args.truncate_by_mask:
                    seq_final = trim_by_mask(seq_raw, batch_msk[i], length_meta)
                else:
                    seq_final = seq_raw
                
                # 清理序列（只保留标准20个氨基酸）
                allowed = set("ACDEFGHIKLMNPQRSTVWY")
                cleaned = "".join([c for c in seq_final if c in allowed])
                
                # 质量评估
                e_count = cleaned.count('E')
                e_ratio = e_count / len(cleaned) * 100 if len(cleaned) > 0 else 0
                
                quality = "good"
                if len(cleaned) == 0:
                    quality = "empty"
                elif len(cleaned) < args.min_length:
                    quality = "too_short"
                elif len(cleaned) > args.max_length:
                    quality = "too_long"
                elif e_ratio > args.max_e_ratio:
                    quality = "high_e"
                
                row = {
                    "id": idx,
                    "length_pred": len(cleaned),
                    "length_meta": int(length_meta) if length_meta is not None else None,
                    "aa_seq": cleaned,
                    "e_ratio": e_ratio,
                    "quality": quality
                }
                out_rows.append(row)
                
                # FASTA格式
                quality_tag = f" quality={quality}" if quality != "good" else ""
                e_tag = f" E={e_ratio:.1f}%" if e_ratio > 0 else ""
                fasta_lines.append(f">{idx} len={len(cleaned)}{quality_tag}{e_tag}\n{cleaned}\n")
            
            # 进度报告
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            if (e % (bs*5) == 0) or (e==N):
                elapsed = time.time() - t0
                rate = e / elapsed if elapsed > 0 else 0
                eta = (N - e) / rate if rate > 0 else 0
                mem_info = f", GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type == "cuda" else ""
                print(f"📊 进度: {e}/{N} ({100*e/N:.1f}%), 速度: {rate:.1f} seq/s, 剩余: {eta:.0f}s{mem_info}")
    
    # 4. 分析结果
    print(f"\n📈 分析结果...")
    sequences = [r["aa_seq"] for r in out_rows]
    analysis = analyze_aa_composition(sequences)
    
    # 生成报告
    report = generate_quality_report(analysis, args)
    
    # 5. 保存结果
    print(f" 保存结果...")
    csv_path, fasta_path, jsonl_path = save_results(out_rows, fasta_lines, analysis, report, args)
    
    # 6. 总结
    dt = time.time() - t0
    total_seqs = len(out_rows)
    valid_seqs = sum(1 for r in out_rows if len(r["aa_seq"]) > 0)
    good_quality = sum(1 for r in out_rows if r["quality"] == "good")
    
    print(f"解码完成!")
    print(f"总耗时: {dt:.1f}s ({total_seqs/dt:.1f} seq/s)")
    print(f"统计: {valid_seqs}/{total_seqs} 有效序列, {good_quality}/{total_seqs} 高质量序列")
    print(f"输出文件:")
    print(f"   - {csv_path}")
    print(f"   - {fasta_path}")
    print(f"   - {jsonl_path}")
    if args.generate_report:
        print(f"   - {args.out_prefix}_analysis_report.txt")
        print(f"   - {args.out_prefix}_analysis.json")
    
    # 7. 显示报告摘要
    if analysis:
        e_ratio = analysis['aa_frequencies'].get('E', 0)
        avg_length = analysis.get('avg_length', 0)
        print(f"质量摘要:")
        print(f"   平均长度: {avg_length:.1f}")
        print(f"   E含量: {e_ratio:.1f}% (天然: 6.8%)")
        
        if e_ratio > 20:
            print(f"   E含量严重偏高，建议优化参数")
        elif e_ratio > 15:
            print(f"   E含量明显偏高，可考虑后处理")
        else:
            print(f"   E含量在可接受范围内")
    
    print(f"查看完整分析报告: cat {args.out_prefix}_analysis_report.txt")

if __name__ == "__main__":
    main()
