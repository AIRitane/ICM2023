import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs


class MyKmeans:
    x_distance = None  # 距离矩阵
    x_center_distance = None  # 中心点距离矩阵

    # 原始数据点绘制
    # 聚类后数据点图片
    # 整体图片绘制

    def __init__(self, Data):
        self.X = Data
        self.fig = plt.figure(figsize=(8, 3))
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        self.colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        # self.ax = self.fig.add_subplot(1, 2, 1)
        self.ax = plt.axes([0.1, 0.1, 0.7, 0.7])
        self.k_means_cluster_centers = None
        self.k_means_labels = None

    def x_draw(self):
        row, _ = np.shape(self.X)
        for i in range(row):
            self.ax.plot(X[i, 0], X[i, 1], '#4EACC5', marker='.')
        self.ax.set_title('Original Data')
        self.ax.set_xticks(())
        self.ax.set_yticks(())

    def x_get_kmeans(self, num):
        k_means = KMeans(init='k-means++', n_clusters=num, n_init=10)
        k_means.fit(self.X)
        x_distance = k_means.transform(self.X)
        self.k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        self.k_means_labels = pairwise_distances_argmin(self.X, self.k_means_cluster_centers)

    def x_show(self):
        plt.show()


# ######################################
# Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

MyKmeans = MyKmeans(X)
MyKmeans.x_draw()
MyKmeans.x_show()

# # compute clustering with K-Means
# k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
# t0 = time.time()
# k_means.fit(X)
# t_batch = time.time() - t0
#
# k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
# k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
#
# # K-means
# ax = fig.add_subplot(1, 2, 2)
# for k, col in zip(range(n_clusters), colors):
#     my_members = k_means_labels == k		# my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
#     cluster_center = k_means_cluster_centers[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
#             markerfacecolor=col, marker='.')	# 将同一类的点表示出来
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', marker='o')	# 将聚类中心单独表示出来
# ax.set_title('KMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
#
# plt.show()
