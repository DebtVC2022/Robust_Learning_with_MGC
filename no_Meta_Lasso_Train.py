import numpy as np
from multiprocessing import Pool
from scipy.stats import norm

from Generate_Noisy_Label_Data import generate_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def probit(x):
    return norm.cdf(x)

def compute_accuracy(X, y, beta):
    y_pred = np.sign(np.dot(X, beta))
    y_pred[y_pred == 0] = 1  # 将预测值为0的样本视为正类
    return np.mean(y_pred == y)

def lasso_derivative(beta, lam):
    return np.where(beta > 0, lam, np.where(beta < 0, -lam, 0))

def loss_gradient_lasso(y, x, beta, gamma, M, lam=1.0):
    z = y * (gamma + 1) * np.dot(beta.T, x)
    weight = (np.exp(z) / (1 + np.exp(z))) ** (gamma / (gamma + 1))

    lasso_grad = lasso_derivative(beta, lam)
    
    gradient = (gamma / (gamma + 1)) * weight * (sigmoid(z) - 1) * y * x + lasso_grad

    return gradient


def logistic_regression(X, noise_indices, y_noisy, y_true, learning_rate, gamma, M, n_epochs, batch_size, early_stop_threshold=0.0001, early_stop_patience=5):
    n_samples, n_features = X.shape
    beta = np.random.rand(n_features)  # 随机初始化beta
    
    best_accuracy_true = 0
    no_improvement_count = 0

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y_noisy[indices]

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            gradient = np.zeros(n_features)

            for xi, yi in zip(X_batch, y_batch):
                # 不带惩罚项、不带元学习的梯度计算
                gradient += loss_gradient_lasso(yi, xi, beta, gamma, M)

            beta += learning_rate * gradient / batch_size

        accuracy_noisy = compute_accuracy(X, y_noisy, beta)
        accuracy_true = compute_accuracy(X, y_true, beta)
        #print(f"Epoch {epoch + 1}, accuracy_noise: {accuracy_noisy:.4f}, accuracy_tru: {accuracy_true:.4f}")

        # 检查accuracy_true是否有提高
        if accuracy_true - best_accuracy_true > early_stop_threshold:
            best_accuracy_true = accuracy_true
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 如果连续early_stop_patience轮次没有提高，则早停训练
        if no_improvement_count >= early_stop_patience:
            #print("Early stopping triggered.")
            break

        # 返回所有权重小于M的标签位置
        weights = np.array([((np.exp(yi * (gamma + 1) * np.dot(beta.T, xi)) / (1 + np.exp((gamma + 1) * np.dot(beta.T, xi)))) ** (gamma / (gamma + 1))) for xi, yi in zip(X, y_noisy)])
        small_weight_indices = np.where(weights < M)[0]
        #print(f"Small weight indices: {small_weight_indices}")

        # 计算权重小于M的标签位置与真实噪声标签位置的一致性
        matching_noise = len(np.intersect1d(small_weight_indices, noise_indices))
        #print(f"Matching noise labels: {matching_noise} out of {len(noise_indices)}")
        
        # 计算三个比例
        ratio_1 = matching_noise / len(small_weight_indices) if len(small_weight_indices) > 0 else 0
        ratio_2 = matching_noise / len(noise_indices)
        ratio_3 = 2 * ratio_1 * ratio_2 / (ratio_1 + ratio_2) if (ratio_1 + ratio_2) > 0 else 0

        #print(f"Ratio 1 (True noise in small weight indices): {ratio_1:.4f}")
        #print(f"Ratio 2 (Detected noise in true noise labels): {ratio_2:.4f}")
        #print(f"Ratio 3 (Harmonic mean of Ratio 1 and 2): {ratio_3:.4f}")

    return beta, accuracy_noisy, accuracy_true, ratio_1, ratio_2, ratio_3


def test_logistic_regression(test_X, test_noise_indices, test_labels_noisy, test_labels_true, trained_beta, gamma, M):
    accuracy_noisy_test = compute_accuracy(test_X, test_labels_noisy, trained_beta)
    accuracy_true_test = compute_accuracy(test_X, test_labels_true, trained_beta)

    #print("Accuracy on noisy test data:", accuracy_noisy_test)
    #print("Accuracy on true test data:", accuracy_true_test)

    # 计算测试集中权重小于 M 的标签位置
    weights_test = np.array([((np.exp(yi * (gamma + 1) * np.dot(trained_beta.T, xi)) / (1 + np.exp((gamma + 1) * np.dot(trained_beta.T, xi)))) ** (gamma / (gamma + 1))) for xi, yi in zip(test_X, test_labels_noisy)])
    small_weight_indices_test = np.where(weights_test < M)[0]

    # 计算权重小于 M 的标签位置与真实噪声标签位置的一致性
    matching_noise_test = len(np.intersect1d(small_weight_indices_test, test_noise_indices))
    #print("Matching noise labels in the test dataset:", matching_noise_test, "out of", len(test_noise_indices))

    # 计算三个比例
    ratio_1_test = matching_noise_test / len(small_weight_indices_test) if len(small_weight_indices_test) > 0 else 0
    ratio_2_test = matching_noise_test / len(test_noise_indices)
    ratio_3_test = 2 * ratio_1_test * ratio_2_test / (ratio_1_test + ratio_2_test) if (ratio_1_test + ratio_2_test) > 0 else 0

    return accuracy_noisy_test, accuracy_true_test, ratio_1_test, ratio_2_test, ratio_3_test

def worker(random_seed):
    X, labels_noisy, labels_true, noise_indices = generate_data(1000, 1500, 0.3, random_seed)  #generate_from_real_data("./real_data/Breast_Cancer_Dataset/breast-cancer_new_train.csv", 0.1, random_seed)
    test_X, test_labels_noisy, test_labels_true, test_noise_indices = generate_data(200, 1500, 0.3, random_seed)    # generate_from_real_data("./real_data/Breast_Cancer_Dataset/breast-cancer_new_train.csv", 0.1, random_seed)

    learning_rate = 0.01
    gamma = 0.5
    M = 0.5
    n_epochs = 100
    batch_size = 32

    trained_beta, accuracy_noisy, accuracy_true, ratio_1, ratio_2, ratio_3 = logistic_regression(X, noise_indices, labels_noisy, labels_true, learning_rate, gamma, M, n_epochs, batch_size)

    accuracy_noisy_test, accuracy_true_test, ratio_1_test, ratio_2_test, ratio_3_test = test_logistic_regression(test_X, test_noise_indices, test_labels_noisy, test_labels_true, trained_beta, gamma, M)

    return accuracy_noisy, accuracy_true, accuracy_noisy_test, accuracy_true_test, ratio_1, ratio_2, ratio_3, ratio_1_test, ratio_2_test, ratio_3_test


if __name__ == "__main__":
    with Pool(processes=12) as pool:
        results = pool.map(worker, range(1101,1150))

    accuracy_noisy_lits, accuracy_true_list, accuracy_noisy_test_lits, accuracy_true_test_list, ratio_1_list, ratio_2_list, ratio_3_list, ratio_1_test_list, ratio_2_test_list, ratio_3_test_list = zip(*results)

    print(np.array(accuracy_noisy_lits).mean())
    print(np.array(accuracy_true_list).mean())
    print(np.array(ratio_1_list).mean())
    print(np.array(ratio_2_list).mean())
    print(np.array(ratio_3_list).mean())

    print(np.array(accuracy_noisy_test_lits).mean())
    print(np.array(accuracy_true_test_list).mean())
    print(np.array(ratio_1_test_list).mean())
    print(np.array(ratio_2_test_list).mean())
    print(np.array(ratio_3_test_list).mean())