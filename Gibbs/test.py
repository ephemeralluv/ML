import numpy as np

# 定义概率模型
model = {'C': [], 'U': ['C'], 'W': ['U'], 'B': ['U'], 'D': ['W', 'B']}

# 定义概率表
cpd_C = np.array([[0.5, 0.5]])
cpd_U = np.array([[0.99, 0.01], [0.05, 0.95]])
cpd_W = np.array([[0.95, 0.05], [0.1, 0.9]])
cpd_B = np.array([[0.99, 0.01], [0.7, 0.3]])
cpd_D = np.array([[[1.0, 0.0], [0.95, 0.05]], [[0.7, 0.3], [0.665, 0.335]]])

# 初始化随机样本
sample = {'C': 0, 'U': 0, 'W': 0, 'B': 0, 'D': 0}

# 定义条件概率表中的值在随机样本中的索引
cpd_indices = {'C': [0, 1], 'U': [0, 1], 'W': [0, 1], 'B': [0, 1], 'D': [0, 1]}

# 定义采样次数和燃烧期次数
n_samples = 10000
burn_in = 1000

# 初始化计数器
count_d_true = 0
count_d_true_u_true_w_true = 0

# 吉布斯采样
for i in range(n_samples + burn_in):
    for node in model:
        parents = model[node]
        if not parents:
            # 如果没有父节点，则直接从概率表中采样
            sample[node] = np.random.choice(cpd_indices[node], p=cpd_C.flatten())
        else:
            # 如果有父节点，则计算条件概率表
            parent_values = tuple(sample[parent] for parent in parents)
            cpd = eval(f'cpd_{node}[parent_values]')
            sample[node] = np.random.choice(cpd_indices[node], p=cpd.flatten())

    # 统计计数器
    if sample['U'] == 1 and sample['W'] == 1:
        count_d_true_u_true_w_true += 1
        if sample['D'] == 1:
            count_d_true += 1

# 计算P(D=1|U=1,W=1)
p_d_true_u_true_w_true = count_d_true / count_d_true_u_true_w_true

print(f'P(D=1|U=1,W=1) = {p_d_true_u_true_w_true:.4f}')
