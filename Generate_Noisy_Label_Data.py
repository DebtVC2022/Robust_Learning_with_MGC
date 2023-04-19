import numpy as np

def generate_data(sample_size, feature_dimension, noise_ratio, random_seed=42):
    # 设置随机数种子以获得可重复的结果
    np.random.seed(random_seed)

    # 设定参数
    #sample_size = 500
    #feature_dimension = 1000
    #noise_ratio = 0.1

    # 生成线性可分数据集
    X = np.random.rand(sample_size, feature_dimension) * 2 - 1  # 均匀分布在[-1, 1]区间
    w = np.random.rand(feature_dimension, 1)  # 权重向量
    y = X @ w  # 线性组合

    # 生成二值标签
    threshold = np.median(y)
    labels = (y > threshold).astype(int).reshape(-1)

    # 复制一份原始的准确标签
    original_labels = labels.copy()

    # 为标签添加噪声
    noise_count = int(sample_size * noise_ratio)
    noise_indices = np.random.choice(sample_size, noise_count, replace=False)

    for i in noise_indices:
        labels[i] = 1 - labels[i]  # 将正确标签翻转为错误标签

    return X, labels, original_labels, noise_indices