import numpy as np


def kmeans(data, k, max_iter=100):
    # 在0到data.shape[0]内，无放回采样k个值
    centers = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iter):
        # 利用广播计算所有点到centers的距离
        # data [n, dim]
        # np.expand_dims(data, axis=1) [n, 1, dim]
        # centers [k, dim]
        # np.expand_dims(data, axis=1) - centers
        # 将np.expand_dims(data, axis=1)广播为[n, k, dim]，相当于把dim扩展k次，以求单个数据点到k个中心的距离
        # 将centers广播为[n, k, dim]，相当于把自身扩展n次，以求与n个数据点的距离
        # np.linalg.norm求L2范数，在axis=2上求L2范数等价于求距离
        distances = np.linalg.norm(np.expand_dims(data, axis=1) - centers, axis=2)
        # distances [n, k] 相当于n个点到k个点的距离
        # 在axis=1上求min，相当于找到每个点距离最近的中心
        labels = np.argmin(distances, axis=1)
        # labels [n]
        # 对于每个簇，找到其对应的所有的点，在axis=0上求该簇的中心
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return labels, centers


data = np.random.rand(100, 2)
k = 3
labels, centers = kmeans(data, k)
print(labels)
print(centers)