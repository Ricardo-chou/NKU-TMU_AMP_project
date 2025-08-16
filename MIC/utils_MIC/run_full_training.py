#!/usr/bin/env python3
"""
完整的两层建模训练脚本
执行完整的训练流程，包括两个模型的训练和评估
"""

from train_discriminators import AMPTrainer
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='训练AMP判别器模型')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停patience')
    parser.add_argument('--device', type=str, default='auto', help='训练设备')
    parser.add_argument('--features_dir', type=str, default='/root/NKU-TMU_AMP_project/features', help='特征目录')
    parser.add_argument('--output_dir', type=str, default='/root/NKU-TMU_AMP_project/model_outputs', help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AMP判别器两层建模训练")
    print("="*60)
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"早停patience: {args.patience}")
    print(f"设备: {args.device}")
    print(f"特征目录: {args.features_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    # 创建训练器
    trainer = AMPTrainer(device=args.device, output_dir=args.output_dir)
    
    # 加载数据
    print("\n步骤1: 加载特征数据")
    start_time = time.time()
    trainer.load_data(features_dir=args.features_dir)
    print(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 训练序列聚合模型
    print("\n" + "="*60)
    print("步骤2: 训练序列聚合回归模型（模型A）")
    print("="*60)
    
    start_time = time.time()
    sequence_model, seq_history = trainer.train_sequence_model(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    seq_training_time = time.time() - start_time
    print(f"序列聚合模型训练完成，耗时: {seq_training_time:.2f}秒")
    
    # 训练条件回归模型
    print("\n" + "="*60)
    print("步骤3: 训练条件回归模型（模型B）")
    print("="*60)
    
    start_time = time.time()
    conditional_model, cond_history = trainer.train_conditional_model(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    cond_training_time = time.time() - start_time
    print(f"条件回归模型训练完成，耗时: {cond_training_time:.2f}秒")
    
    # 生成训练报告
    print("\n" + "="*60)
    print("步骤4: 生成训练报告")
    print("="*60)
    
    try:
        trainer.plot_training_curves()
        print("训练曲线已保存")
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")
    
    # 总结
    total_time = seq_training_time + cond_training_time
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"序列聚合模型最佳验证损失: {seq_history['best_val_loss']:.4f}")
    print(f"条件回归模型最佳验证损失: {cond_history['best_val_loss']:.4f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"模型和结果已保存到: {args.output_dir}")
    print("="*60)
    
    # 生成最终报告文件
    report_path = f"{args.output_dir}/training_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AMP判别器训练报告\n\n")
        f.write("## 训练配置\n")
        f.write(f"- 批次大小: {args.batch_size}\n")
        f.write(f"- 学习率: {args.learning_rate}\n")
        f.write(f"- 训练轮数: {args.num_epochs}\n")
        f.write(f"- 早停patience: {args.patience}\n")
        f.write(f"- 设备: {args.device}\n\n")
        
        f.write("## 训练结果\n")
        f.write(f"- 序列聚合模型最佳验证损失: {seq_history['best_val_loss']:.4f}\n")
        f.write(f"- 条件回归模型最佳验证损失: {cond_history['best_val_loss']:.4f}\n")
        f.write(f"- 序列聚合模型训练时间: {seq_training_time:.2f}秒\n")
        f.write(f"- 条件回归模型训练时间: {cond_training_time:.2f}秒\n")
        f.write(f"- 总训练时间: {total_time:.2f}秒\n\n")
        
        f.write("## 文件说明\n")
        f.write("- `sequence_regression_best.pt`: 序列聚合模型权重\n")
        f.write("- `conditional_regression_best.pt`: 条件回归模型权重\n")
        f.write("- `sequence_training_history.pkl`: 序列模型训练历史\n")
        f.write("- `conditional_training_history.pkl`: 条件模型训练历史\n")
        f.write("- `sequence_test_results.pkl`: 序列模型测试结果\n")
        f.write("- `conditional_test_results.pkl`: 条件模型测试结果\n")
        f.write("- `training_curves.png`: 训练曲线图\n")
    
    print(f"训练报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
