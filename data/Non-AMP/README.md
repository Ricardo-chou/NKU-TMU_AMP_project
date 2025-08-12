# 负样本生成脚本

这个脚本用于从UniProtKB的FASTA文件中生成抗菌肽（AMP）的负样本集。

## 功能特点

1. **分析正样本长度分布**: 分析`final_AMP.csv`中正样本的长度分布
2. **智能负样本生成**: 根据正样本的长度分布，从UniProtKB蛋白质中随机截取片段
3. **去重处理**: 自动跳过与正样本重复的序列
4. **可视化分析**: 生成正样本长度分布图表
5. **质量控制**: 确保负样本的质量和多样性

## 文件结构

```
Non-AMP/
├── generate_negative_samples.py    # 主脚本
├── requirements.txt                # 依赖包列表
├── README.md                      # 说明文档
├── uniprot_sprot.fasta           # UniProtKB FASTA文件
└── Non_AMP_UniProtKB.csv        # 生成的负样本集（运行后生成）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保以下文件存在：
   - `../AMP/final_AMP.csv` (正样本文件)
   - `uniprot_sprot.fasta` (UniProtKB FASTA文件)

2. 运行脚本：
   ```bash
   python generate_negative_samples.py
   ```

## 输出文件

- `Non_AMP_UniProtKB.csv`: 包含负样本序列的CSV文件
- `positive_length_distribution.png`: 正样本长度分布图表

## 算法说明

1. **长度分布分析**: 统计正样本的长度分布，用于指导负样本生成
2. **随机截取**: 从UniProtKB蛋白质中随机选择起始位置和长度
3. **智能长度选择**: 基于正样本长度分布的概率权重选择目标长度
4. **去重检查**: 确保生成的负样本不与正样本重复
5. **质量控制**: 设置合理的长度范围（5-200氨基酸）

## 参数说明

- `num_samples`: 生成的负样本数量（默认与正样本数量相同）
- `min_length`: 最小序列长度（默认5）
- `max_length`: 最大序列长度（默认200）

## 注意事项

- 脚本会自动处理FASTA文件格式
- 生成的负样本会保持与正样本相似的长度分布
- 建议在运行前备份重要数据
- 对于大型FASTA文件，处理时间可能较长
