import numpy as np

def gibbs_sampling(model, cpds, n_samples=10000, burn_in=1000):
    """
    对给定的概率图模型和概率表执行吉布斯采样，并返回特定条件下的概率值。

    参数：
    - model：dict，表示概率模型的有向无环图。每一项为 mode:[parents]
    - cpds，Conditional Probability Distribution（条件概率分布）：dict，表示每个节点的条件概率表。
        例：
            'U': [
                    [P(W|U=1), ~P(W|U=1)],
                    [P(W|U=0), ~P(W|U=0)]
                  ]
    - n_samples：int，采样次数，默认为10000。
    - burn_in：int，燃烧期次数，默认为1000。

    返回：
    - float，P(D=1|U=1,W=1)的值。
    """

    # 初始化初始状态，随机采样：sample = {'C': ？, 'U': ？, 'W': ?, 'B': ?, 'D': ?}
    sample = {}
    for node in model:
        sample[node] = np.random.randint(0, 2)

    # 初始化计数器
    count_d_true = 0
    count_d_true_u_true_w_true = 0

    # 吉布斯采样
    for i in range(n_samples + burn_in):
        # gibbs采样一次，更新每个结点的1/0状态
        for node in model:
            parents = model[node]
            if not parents:
                # 如果没有父节点，则直接从概率表中采样
                sample[node] = np.random.randint(len(cpds[node]))
            else:
                # 如果有父节点，则计算条件概率表
                parent_values = tuple(sample[parent] for parent in parents) # 父节点的取值
                cpd = cpds[node][parent_values] # cpds[node]找到当前结点条件概率表，[parent_values]，对应条件概率p(node|parent)
                sample[node] = np.random.choice(2, p=cpd) # 从p(node|parent)分布中随机取一个0/1元素作为新的状态

        # 统计计数器
        if sample['U'] == 1 and sample['W'] == 1:
            count_d_true_u_true_w_true += 1
            if sample['D'] == 1:
                count_d_true += 1

    # 计算P(D=1|U=1,W=1)
    p_d_true_u_true_w_true = count_d_true / count_d_true_u_true_w_true

    return p_d_true_u_true_w_true

if __name__ == '__main__':
    model = {'C': [], 'U': ['C'], 'W': ['U'], 'B': ['U'], 'D': ['W', 'B']}
    cpds = {'C': np.array([[0.5, 0.5]]),
            'U': np.array([[0.90, 0.10], [0.05, 0.95]]),
            'W': np.array([[0.95, 0.05], [0.1, 0.9]]),
            'B': np.array([[0.99, 0.01], [0.7, 0.3]]),
            'D': np.array([[[1.0, 0.0], [0.95, 0.05]], [[0.7, 0.3], [0.665, 0.335]]])}

    p = gibbs_sampling(model, cpds)
    print(f'P(D=1|U=1,W=1) = {gibbs_sampling(model, cpds):.4f}')









