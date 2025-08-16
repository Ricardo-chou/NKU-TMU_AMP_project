# AMP判别器训练报告

## 训练配置
- 批次大小: 128
- 学习率: 0.0002
- 训练轮数: 50
- 早停patience: 15
- 设备: auto

## 训练结果
- 序列聚合模型最佳验证损失: 0.0725
- 条件回归模型最佳验证损失: 0.1865
- 序列聚合模型训练时间: 10.03秒
- 条件回归模型训练时间: 40.16秒
- 总训练时间: 50.20秒

## 文件说明
- `sequence_regression_best.pt`: 序列聚合模型权重
- `conditional_regression_best.pt`: 条件回归模型权重
- `sequence_training_history.pkl`: 序列模型训练历史
- `conditional_training_history.pkl`: 条件模型训练历史
- `sequence_test_results.pkl`: 序列模型测试结果
- `conditional_test_results.pkl`: 条件模型测试结果
- `training_curves.png`: 训练曲线图
