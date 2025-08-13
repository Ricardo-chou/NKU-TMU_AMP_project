"""
ProT-Diff: ProtT5 编码器 + 连续扩散模型(Trans-UNet风格) + ProtT5 解码器
目的：在 ProtT5 latent 空间 (48,1024) 训练扩散模型，采样并通过 ProtT5 解码得到候选序列。

使用说明：
1) 安装依赖：
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 根据你的CUDA版本调整
   pip install transformers accelerate peft einops datasets numpy pandas scikit-learn tqdm

2) 准备数据：
   - 传入一个 FASTA/CSV 列表或 python list[str]，每条为已过滤的肽序列（仅含大写20 AA，长度 5–48）。
   - 本脚本会用 ProtT5 编码成 (L,1024)，再 0-padding 到 (48,1024)。
   - 你也可以将 embeddings 预先缓存为 .npy 以加速训练（见 TODO）。

3) 训练：
   - 主要超参与论文一致：latent 形状(48,1024)，训练步数设置、sqrt 噪声日程、预测 x0。
   - 模型是 1D U-Net + 自注意力瓶颈，近似 Trans-UNet。

4) 采样：
   - 200 步 DDPM 采样，可选 Gaussian / Uniform 噪声（数据少时用 Uniform 增强多样性）。
   - 采样得到 (48,1024) latent，经“去零行”截断，再送入 ProtT5 解码器生成氨基酸序列。

5) 输出：
   - 保存生成序列到 generated_seqs.csv，后续交由判别器与理化规则筛选。

警告：
- ProtT5-XL 模型很大（~3B参数）。若显存吃紧，考虑使用 half-precision、CPU offload、或改用 smaller encoder（如 ProtT5-base）。
- 本文件为复现骨架，工程化时建议将编码/训练/采样拆分脚本，并加入断点续训、混合精度、日志记录等。
"""
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from einops import rearrange

# ================================
# 配置
# ================================
@dataclass
class Config:
    max_len: int = 48
    embed_dim: int = 1024
    batch_size: int = 8
    lr: float = 1e-4
    num_workers: int = 2
    epochs: int = 3
    # 扩散相关
    train_diffusion_steps: int = 2000  # 训练时间步（论文设定）
    sample_steps: int = 200            # 采样下采样步数
    beta_schedule: str = "sqrt"        # Diffusion-LM 的 sqrt schedule 近似
    predict_x0: bool = True            # 预测 x0
    uniform_noise_sampling: bool = False
    
    # 模型宽度
    unet_channels: int = 256
    attn_dim: int = 256
    attn_heads: int = 8
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ================================
# ProtT5 编码与解码封装
# ================================
class ProtT5Wrapper:
    def __init__(self, device=CFG.device, half=True):
        # 使用 Rostlab/prot_t5_xl_uniref50
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.decoder = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.encoder.eval().to(device)
        self.decoder.eval().to(device)
        if half and device == "cuda":
            self.encoder.half()
            self.decoder.half()
        self.device = device

    @torch.no_grad()
    def encode(self, seq: str, max_len=CFG.max_len) -> torch.Tensor:
        """将 AA 序列编码为 (max_len, 1024) 的 embedding，并做 0-padding。
        输入序列应仅包含 20 AA 字母且长度 ≤ max_len。
        """
        seq = seq.replace(" ", "")
        # prot_t5 期望氨基酸之间有空格分隔
        spaced = " ".join(list(seq))
        batch = self.tokenizer([spaced], return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        emb = self.encoder(**batch).last_hidden_state.squeeze(0)  # (L+special, 1024)
        # 去掉起始/结束特殊符（通常第一和最后一个为特殊符）
        if emb.size(0) >= 2:
            emb = emb[1:-1]
        L = emb.size(0)
        if L > max_len:
            emb = emb[:max_len]
            L = max_len
        pad = torch.zeros(max_len - L, emb.size(1), device=emb.device, dtype=emb.dtype)
        out = torch.cat([emb, pad], dim=0)  # (max_len, 1024)
        return out

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, max_len=CFG.max_len) -> str:
        """将 (T,1024) latent 作为 encoder 输出，调用 T5 解码器生成 AA 序列。"""
        # 去掉主要为 0 的行（padding 截断）
        with torch.no_grad():
            row_norm = latents.float().abs().mean(dim=1)
            mask = row_norm > (row_norm.mean() * 0.1)
            if mask.any():
                latents = latents[mask]
            latents = latents.unsqueeze(0)  # (1, L, 1024)
            enc_out = BaseModelOutput(last_hidden_state=latents.to(self.device))
            # 解码：贪心/beam 均可，这里用贪心以保持速度和一致性
            gen_ids = self.decoder.generate(
                encoder_outputs=enc_out,
                max_length=min(max_len, latents.size(1) + 2),
                num_beams=1,
                do_sample=False,
                decoder_start_token_id=self.decoder.config.decoder_start_token_id or 0,
                eos_token_id=self.decoder.config.eos_token_id,
            )
            text = self.decoder.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            # 去除空格，确保返回 AA 字母
            aa = text.replace(" ", "").replace("▁", "")
            # 只保留 20 AA
            aa = ''.join([c for c in aa if c in "ACDEFGHIKLMNPQRSTVWY"])
            return aa[:max_len]

# ================================
# 数据集
# ================================
class PeptideDataset(Dataset):
    def __init__(self, sequences: List[str], encoder: ProtT5Wrapper):
        self.sequences = sequences
        self.enc = encoder

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        emb = self.enc.encode(seq)  # (48,1024)
        return emb.float()

# ================================
# 扩散调度器（sqrt schedule 近似）
# ================================
class DiffusionScheduler:
    def __init__(self, T=CFG.train_diffusion_steps, schedule="sqrt"):
        self.T = T
        if schedule == "sqrt":
            # 参考 Diffusion-LM，给出 sqrt 形状的 beta
            t = torch.linspace(0, 1, T+1)
            betas = (torch.sqrt(t[1:]) - torch.sqrt(t[:-1])).clamp(1e-6, 0.02)
        else:
            betas = torch.linspace(1e-4, 0.02, T)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

# ================================
# Trans-UNet 风格 1D 模型（近似实现）
# ================================
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv1d(c, c, 3, padding=1), nn.GroupNorm(8, c)
        )
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.net(x) + x)

class SelfAttention1D(nn.Module):
    def __init__(self, c, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
    def forward(self, x):  # x: (B,C,L) -> (B,L,C)
        x_l = rearrange(x, 'b c l -> b l c')
        x_n = self.norm(x_l)
        y, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        y = x_l + y
        return rearrange(y, 'b l c -> b c l')

class TimeEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, c*4), nn.SiLU(), nn.Linear(c*4, c)
        )
    def forward(self, t_embed):
        return self.mlp(t_embed)

def sinusoidal_embedding(t, dim=256):
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1e-4), math.log(1.0), half, device=device)
    )
    ang = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class UNet1D(nn.Module):
    def __init__(self, in_dim=CFG.embed_dim, base_c=CFG.unet_channels, heads=CFG.attn_heads):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, base_c, 1)
        self.time = TimeEmbedding(base_c)

        self.down1 = nn.Sequential(ResBlock(base_c), ResBlock(base_c))
        self.down2 = nn.Sequential(nn.Conv1d(base_c, base_c*2, 4, stride=2, padding=1), ResBlock(base_c*2))
        self.down3 = nn.Sequential(nn.Conv1d(base_c*2, base_c*4, 4, stride=2, padding=1), ResBlock(base_c*4))

        self.mid_attn = SelfAttention1D(base_c*4, heads=heads)
        self.mid_res = ResBlock(base_c*4)

        self.up3 = nn.Sequential(nn.ConvTranspose1d(base_c*4, base_c*2, 4, stride=2, padding=1), ResBlock(base_c*2))
        self.up2 = nn.Sequential(nn.ConvTranspose1d(base_c*2, base_c, 4, stride=2, padding=1), ResBlock(base_c))
        self.up1 = nn.Sequential(ResBlock(base_c), ResBlock(base_c))

        self.proj_out = nn.Conv1d(base_c, in_dim, 1)

    def forward(self, x, t):  # x: (B,L,1024)
        x = rearrange(x, 'b l c -> b c l')
        t_embed = sinusoidal_embedding(t, x.size(1))  # (B, C)
        t_embed = self.time(t_embed)
        t_embed = t_embed[..., None]  # (B,C,1)

        x = self.proj_in(x)
        x = x + t_embed

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        m = self.mid_attn(d3)
        m = self.mid_res(m)

        u3 = self.up3(m) + d2
        u2 = self.up2(u3) + d1
        u1 = self.up1(u2)

        out = self.proj_out(u1)
        out = rearrange(out, 'b c l -> b l c')
        return out

# ================================
# 扩散训练/采样
# ================================
class DiffusionModel(nn.Module):
    def __init__(self, net: nn.Module, scheduler: DiffusionScheduler, predict_x0=True):
        super().__init__()
        self.net = net
        self.sch = scheduler
        self.predict_x0 = predict_x0

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sch.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_om = self.sch.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_ac * x0 + sqrt_om * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.net(xt, t)
        if self.predict_x0:
            target = x0
        else:
            target = noise
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, n, device, steps=CFG.sample_steps, noise_type: str = "gaussian"):
        T = self.sch.T
        # 选择等间隔的时间步
        step_idx = torch.linspace(T-1, 0, steps, dtype=torch.long)
        x = torch.randn(n, CFG.max_len, CFG.embed_dim, device=device)
        for i in tqdm(step_idx, desc="sampling"):
            i = i.item()
            t = torch.full((n,), int(i), device=device, dtype=torch.long)
            # 预测 x0
            pred_x0 = self.net(x, t)
            beta = self.sch.betas[i]
            alpha = self.sch.alphas[i]
            alpha_cum = self.sch.alphas_cumprod[i]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha)

            # 逆扩散一步（DDPM 简化版）
            eps = (x - torch.sqrt(alpha_cum) * pred_x0) / torch.sqrt(1 - alpha_cum + 1e-8)
            mean = (1/torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - alpha_cum + 1e-8) * eps)
            if i > 0:
                if noise_type == "uniform":
                    z = torch.empty_like(x).uniform_(-1, 1)
                else:
                    z = torch.randn_like(x)
                x = mean + sqrt_one_minus_alpha * z
            else:
                x = mean
        return x

# ================================
# 训练与采样管线
# ================================
def train_diffusion(seqs: List[str], save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    prot = ProtT5Wrapper(device=CFG.device, half=True)
    ds = PeptideDataset(seqs, prot)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)

    sch = DiffusionScheduler(T=CFG.train_diffusion_steps, schedule=CFG.beta_schedule).to(CFG.device)
    net = UNet1D(in_dim=CFG.embed_dim).to(CFG.device)
    diff = DiffusionModel(net, sch, predict_x0=CFG.predict_x0).to(CFG.device)
    opt = torch.optim.AdamW(diff.parameters(), lr=CFG.lr)

    global_step = 0
    for epoch in range(CFG.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{CFG.epochs}")
        for x0 in pbar:
            x0 = x0.to(CFG.device)  # (B,48,1024)
            t = torch.randint(0, sch.T, (x0.size(0),), device=CFG.device, dtype=torch.long)
            loss = diff.p_losses(x0, t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            opt.step()
            global_step += 1
            pbar.set_postfix(loss=float(loss))
        # 每个 epoch 存一次
        torch.save({
            'model': diff.state_dict(),
            'cfg': CFG.__dict__,
        }, os.path.join(save_dir, f"diffusion_epoch{epoch+1}.pt"))
    return prot, diff

@torch.no_grad()
def generate_sequences(prot: ProtT5Wrapper, diff: DiffusionModel, n: int = 500, noise_type: Optional[str] = None):
    if noise_type is None:
        noise_type = "uniform" if CFG.uniform_noise_sampling else "gaussian"
    latents = diff.sample(n=n, device=CFG.device, steps=CFG.sample_steps, noise_type=noise_type)
    seqs = []
    for i in tqdm(range(n), desc="decode"):
        lat = latents[i].detach().float().cpu()
        seq = prot.decode(lat)
        seqs.append(seq)
    return seqs

if __name__ == "__main__":
    # ===== 示例：从简单列表训练与采样 =====
    # TODO: 用你的训练集替换下面示例序列（要求长度 5-48，仅含 20 AA）
    toy_train = [
        "GIGKFLKKAKKFGKAFVKILKK",
        "KKLFKKILKYL",
        "GLFDIVKKVVGAL",
        "RWKIFKKIERVGQHTRDAT",
        "KWKLFKKIPKFLHLAKKF"
    ]
    amp_path = "data/AMP/final_AMP.csv"
    train_sequences = pd.read_csv(amp_path)['sequence'].tolist()
    prot, diff = train_diffusion(train_sequences, save_dir="checkpoints")
    cand = generate_sequences(prot, diff, n=100, noise_type=None)
    pd.Series(cand).to_csv("generated_seqs.csv", index=False, header=["seq"]) 
    print("Saved to generated_seqs.csv, n=", len(cand))
