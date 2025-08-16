#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed version of ProT-Diff latent embeddings decoder with proper amino acid token handling.
"""

import os, sys, json, math, time, argparse, traceback
from typing import List, Optional
import torch
import torch.nn as nn

# HF Transformers
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

def parse_args():
    p = argparse.ArgumentParser(description="Decode ProtT5 latent embeddings to AA sequences.")
    p.add_argument("--pt_path", type=str, required=True,
                   help="Path to generated_embeddings.pt")
    p.add_argument("--model_dir", type=str, default="/root/autodl-tmp/prot_t5_xl_uniref50",
                   help="Local dir of ProtT5-XL-UniRef50 (HF format).")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for generation.")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"],
                   help="Where to run. 'auto' prefers CUDA if available.")
    p.add_argument("--fp16", action="store_true", help="Enable float16 autocast on CUDA.")
    p.add_argument("--max_new_tokens", type=int, default=48,
                   help="Upper bound for decoder generation steps.")
    p.add_argument("--num_beams", type=int, default=1, help="Beam size (1 = greedy).")
    p.add_argument("--temperature", type=float, default=0.8, help="Softmax temp for sampling.")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus sampling.")
    p.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0=off).")
    p.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (1.0=off).")
    p.add_argument("--truncate_by_mask", action="store_true",
                   help="Post-process by mask/lengths to trim padding.")
    p.add_argument("--filter_near_zero", action="store_true", default=True,
                   help="Filter near-zero embeddings for stability (default: True).")
    p.add_argument("--near_zero_threshold", type=float, default=1e-6,
                   help="Threshold for near-zero embedding detection.")
    p.add_argument("--out_prefix", type=str, default="decoded",
                   help="Output file prefix.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging.")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit number of samples to process (for testing).")
    return p.parse_args()

def smart_device(arg: str) -> torch.device:
    if arg == "cpu": return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pt(pt_path: str):
    obj = torch.load(pt_path, map_location="cpu")
    # Required keys
    for k in ["embeddings", "masks"]:
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {pt_path}")
    return obj

def create_aa_token_map(tokenizer):
    """创建氨基酸到token ID的正确映射"""
    aa_chars = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_id = {}
    id_to_aa = {}
    
    # ProtT5使用▁前缀的token
    for aa in aa_chars:
        token_with_prefix = f"▁{aa}"
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_with_prefix)
            if token_id != tokenizer.unk_token_id:  # 确保不是UNK
                aa_to_id[aa] = token_id
                id_to_aa[token_id] = aa
        except:
            pass
    
    print(f"[INFO] Created AA token mapping for {len(aa_to_id)} amino acids")
    if len(aa_to_id) < 20:
        print(f"[WARN] Only found {len(aa_to_id)}/20 amino acid tokens")
    
    return aa_to_id, id_to_aa

def decode_ids_to_sequence(token_ids: List[int], tokenizer) -> str:
    """将token IDs解码为氨基酸序列"""
    sequence = []
    for token_id in token_ids:
        # 跳过PAD token（通常是第一个token）
        if token_id == tokenizer.pad_token_id:
            continue
        # 遇到EOS token停止
        if token_id == tokenizer.eos_token_id:
            break
        
        # 将token ID转换为token字符串
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        
        # 检查是否是氨基酸token（格式：▁X，其中X是氨基酸字母）
        if token_str.startswith('▁') and len(token_str) == 2:
            aa_char = token_str[1:]  # 去掉▁前缀
            if aa_char in "ACDEFGHIKLMNPQRSTVWY":
                sequence.append(aa_char)
        # 忽略其他特殊token
    
    return "".join(sequence)

def build_encoder_outputs(batch_embeds: torch.Tensor, attention_mask: torch.Tensor):
    """
    Wrap latent embeddings as encoder outputs for T5 decoder.
    """
    return BaseModelOutput(
        last_hidden_state=batch_embeds,
    )

def filter_near_zero_embeddings(embeds: torch.Tensor, masks: torch.Tensor, threshold: float = 1e-6, verbose: bool = False) -> torch.Tensor:
    """对近零行进行稳健裁剪，提升解码稳定性"""
    # 计算每个位置的embedding向量的L2范数
    embed_norms = torch.norm(embeds, dim=-1)  # [B, L]
    
    # 识别近零位置（norm < threshold）
    near_zero = embed_norms < threshold
    
    # 更新mask：原本有效且非近零的位置保持有效
    updated_masks = masks & (~near_zero)
    
    # 统计过滤信息
    original_valid = masks.sum().item()
    filtered_valid = updated_masks.sum().item()
    filtered_count = original_valid - filtered_valid
    
    if verbose and filtered_count > 0:
        print(f"[DEBUG] Filtered {filtered_count}/{original_valid} near-zero positions (threshold={threshold})")
    
    # 确保每个序列至少有一个有效位置
    valid_counts = updated_masks.sum(dim=1)  # [B]
    restored_count = 0
    for i, count in enumerate(valid_counts):
        if count == 0:
            # 如果全部被过滤，保留原始mask中的第一个True位置
            first_valid = masks[i].nonzero(as_tuple=True)[0]
            if len(first_valid) > 0:
                updated_masks[i, first_valid[0]] = True
                restored_count += 1
    
    if verbose and restored_count > 0:
        print(f"[DEBUG] Restored {restored_count} sequences that were completely filtered")
    
    return updated_masks

def decode_batch(
    model, tokenizer, device, embeds: torch.Tensor, masks: torch.Tensor,
    args, enable_near_zero_filter: bool = True
) -> List[str]:
    """
    embeds: [B, 48, 1024] float32 on device
    masks : [B, 48]       bool on device
    """
    # 可选：过滤近零embeddings
    if enable_near_zero_filter:
        threshold = getattr(args, 'near_zero_threshold', 1e-6)
        verbose = getattr(args, 'verbose', False)
        masks = filter_near_zero_embeddings(embeds, masks, threshold, verbose)
    
    # 使用采样策略而不是贪婪搜索
    do_sample = args.temperature != 1.0 or args.top_p < 1.0 or args.top_k > 0
    
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else 1.0,
        top_p=args.top_p if do_sample else 1.0,
        top_k=args.top_k if args.top_k > 0 else None,
        repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1.0 else None,
        attention_mask=masks,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # 移除early_stopping，因为我们使用num_beams=1
        length_penalty=1.0,
        no_repeat_ngram_size=2 if args.num_beams > 1 else 0
    )
    
    # 确保decoder配置正确
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    
    # Build encoder outputs with attention mask
    enc_out = build_encoder_outputs(embeds, masks)
    
    try:
        # AMP only if CUDA + fp16 flag
        if device.type == "cuda" and args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_ids = model.generate(encoder_outputs=enc_out, **gen_kwargs)
        else:
            out_ids = model.generate(encoder_outputs=enc_out, **gen_kwargs)
            
        # 使用自定义解码函数
        sequences = []
        for i in range(out_ids.shape[0]):
            seq = decode_ids_to_sequence(out_ids[i].tolist(), tokenizer)
            sequences.append(seq)
        
        return sequences
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise

def trim_by_mask(seq: str, mask_row: torch.Tensor, length_meta: Optional[int]) -> str:
    """
    1) If length_meta provided and sane, cut to that长度；
    2) Else用mask里 True 的数目截断；
    （保守做法：先按 length_meta，其次按 mask）
    """
    if length_meta is not None and length_meta > 0 and length_meta <= len(seq):
        return seq[:length_meta]
    valid_len = int(mask_row.sum().item())
    if 0 < valid_len <= len(seq):
        return seq[:valid_len]
    return seq

def validate_inputs(embeds: torch.Tensor, masks: torch.Tensor, lengths=None):
    """验证输入数据的有效性"""
    if embeds.ndim != 3:
        raise ValueError(f"Expected embeddings to be 3D, got {embeds.ndim}D")
    if embeds.shape[1] != 48:
        raise ValueError(f"Expected sequence length 48, got {embeds.shape[1]}")
    if embeds.shape[2] != 1024:
        raise ValueError(f"Expected embedding dim 1024, got {embeds.shape[2]}")
    if masks.shape[:2] != embeds.shape[:2]:
        raise ValueError(f"Mask shape {masks.shape} doesn't match embeddings {embeds.shape}")
    
    # 检查数据类型
    if embeds.dtype != torch.float32:
        print(f"[WARN] Converting embeddings from {embeds.dtype} to float32")
        embeds = embeds.float()
    if masks.dtype != torch.bool:
        print(f"[WARN] Converting masks from {masks.dtype} to bool")
        masks = masks.bool()
    
    # 检查数据范围
    if torch.isnan(embeds).any():
        print("[WARN] Found NaN values in embeddings")
    if torch.isinf(embeds).any():
        print("[WARN] Found Inf values in embeddings")
    
    # 检查mask的有效性
    valid_mask_counts = masks.sum(dim=1)
    if (valid_mask_counts == 0).any():
        print("[WARN] Found sequences with no valid positions in mask")
    
    return embeds, masks

def main():
    args = parse_args()
    t0 = time.time()
    device = smart_device(args.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Generation config: beams={args.num_beams}, temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

    try:
        # 1) Load .pt
        print(f"[INFO] Loading embeddings from: {args.pt_path}")
        pack = load_pt(args.pt_path)
        embeds = pack["embeddings"]  # [N,48,1024] float32
        masks  = pack["masks"]       # [N,48] bool
        lengths = pack.get("lengths", None)

        # 限制样本数量（用于测试）
        if args.n_samples is not None:
            embeds = embeds[:args.n_samples]
            masks = masks[:args.n_samples]
            if lengths is not None:
                lengths = lengths[:args.n_samples]
            print(f"[INFO] Limited to {args.n_samples} samples for testing")

        # 验证输入数据
        embeds, masks = validate_inputs(embeds, masks, lengths)
        N = embeds.shape[0]
        print(f"[INFO] Loaded N={N} samples, embeddings shape: {embeds.shape}")

        # 2) Load local ProtT5
        print(f"[INFO] Loading ProtT5 from: {args.model_dir}")
        if not os.path.exists(args.model_dir):
            raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
        
        # 修复tokenizer加载问题
        try:
            tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=False)
        except Exception as e1:
            try:
                tokenizer = T5Tokenizer.from_pretrained(args.model_dir, legacy=True)
            except Exception as e2:
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir, legacy=False)
        
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        
        # 创建氨基酸token映射
        aa_to_id, id_to_aa = create_aa_token_map(tokenizer)
        
        # 检查模型配置
        print(f"[INFO] Model config - vocab_size: {model.config.vocab_size}, d_model: {model.config.d_model}")
        print(f"[INFO] Tokenizer - vocab_size: {tokenizer.vocab_size}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"[INFO] Found {len(aa_to_id)} amino acid tokens")
        
        model = model.to(device)
        model.eval()
        
        # 内存优化：如果使用CUDA，启用内存高效的attention
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[INFO] CUDA memory before loading: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    except Exception as e:
        print(f"[ERROR] Failed to load model or data: {e}")
        sys.exit(1)

    # 3) Iterate in batches
    bs = args.batch_size
    out_rows = []
    fasta_lines = []
    jsonl_path = f"{args.out_prefix}.jsonl"
    csv_path   = f"{args.out_prefix}.csv"
    fasta_path = f"{args.out_prefix}.fasta"

    # If lengths is tensor, fetch to CPU list
    lengths_list = None
    if lengths is not None and torch.is_tensor(lengths):
        lengths_list = lengths.tolist()

    # Open JSONL for streaming write
    fjson = open(jsonl_path, "w", encoding="utf-8")

    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s+bs, N)
            batch_emb = embeds[s:e].to(device, non_blocking=True)
            batch_msk = masks[s:e].to(device, non_blocking=True)

            try:
                seqs = decode_batch(model, tokenizer, device, batch_emb, batch_msk, args, enable_near_zero_filter=args.filter_near_zero)
            except Exception as ex:
                print(f"[WARN] Generation failed on batch {s}:{e}, error: {ex}", file=sys.stderr)
                print(f"[INFO] Retrying with conservative settings...", file=sys.stderr)
                
                # 创建保守的fallback参数
                class FallbackArgs:
                    def __init__(self, original_args):
                        self.max_new_tokens = original_args.max_new_tokens
                        self.num_beams = 1
                        self.temperature = 0.8  # 使用轻度采样
                        self.top_p = 0.95
                        self.top_k = 0
                        self.repetition_penalty = 1.2  # 添加重复惩罚
                        self.fp16 = False  # 禁用fp16以提高稳定性
                        self.filter_near_zero = False
                        self.verbose = getattr(original_args, 'verbose', False)
                
                try:
                    fallback_args = FallbackArgs(args)
                    seqs = decode_batch(model, tokenizer, device, batch_emb, batch_msk, fallback_args, enable_near_zero_filter=False)
                    print(f"[INFO] Fallback generation succeeded for batch {s}:{e}")
                except Exception as ex2:
                    print(f"[ERROR] Fallback generation also failed: {ex2}", file=sys.stderr)
                    # 最后的兜底：生成空序列
                    batch_size = batch_emb.shape[0]
                    seqs = [""] * batch_size
                    print(f"[WARN] Using empty sequences for batch {s}:{e}")

            for i, seq in enumerate(seqs):
                idx = s + i
                seq_raw = seq
                length_meta = lengths_list[idx] if lengths_list is not None else None

                if args.truncate_by_mask:
                    seq_final = trim_by_mask(seq_raw, batch_msk[i], length_meta)
                else:
                    seq_final = seq_raw

                # 仅保留标准 20 AA
                allowed = set("ACDEFGHIKLMNPQRSTVWY")
                cleaned = "".join([c for c in seq_final if c in allowed])

                row = {
                    "id": idx,
                    "length_pred": len(cleaned),
                    "length_meta": int(length_meta) if length_meta is not None else None,
                    "aa_seq": cleaned
                }
                out_rows.append(row)
                fjson.write(json.dumps(row, ensure_ascii=False)+"\n")

                fasta_lines.append(f">{idx} len_pred={row['length_pred']} len_meta={row['length_meta']}\n{cleaned}\n")

            # 内存管理和进度报告
            if device.type == "cuda":
                torch.cuda.empty_cache()  # 定期清理GPU缓存
            
            if (e % (bs*5) == 0) or (e==N):  # 更频繁的进度报告
                elapsed = time.time() - t0
                rate = e / elapsed if elapsed > 0 else 0
                eta = (N - e) / rate if rate > 0 else 0
                mem_info = f", GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type == "cuda" else ""
                print(f"[INFO] Decoded {e}/{N} ({100*e/N:.1f}%), rate: {rate:.1f} seq/s, ETA: {eta:.1f}s{mem_info}")

    fjson.close()
    
    # 统计信息
    total_seqs = len(out_rows)
    valid_seqs = sum(1 for r in out_rows if len(r["aa_seq"]) > 0)
    avg_length = sum(r["length_pred"] for r in out_rows) / total_seqs if total_seqs > 0 else 0
    print(f"[INFO] Generation summary: {valid_seqs}/{total_seqs} valid sequences, avg length: {avg_length:.1f}")

    try:
        # 写 CSV/FASTA
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","length_pred","length_meta","aa_seq"])
            for r in out_rows:
                w.writerow([r["id"], r["length_pred"], r["length_meta"], r["aa_seq"]])

        with open(fasta_path, "w", encoding="utf-8") as f:
            f.writelines(fasta_lines)

        dt = time.time()-t0
        rate = total_seqs / dt if dt > 0 else 0
        print(f"[DONE] Successfully wrote:\n  - {csv_path} ({total_seqs} sequences)\n  - {fasta_path}\n  - {jsonl_path}\nTotal time: {dt:.1f}s ({rate:.1f} seq/s)")
        
    except Exception as e:
        print(f"[ERROR] Failed to write output files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
