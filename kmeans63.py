import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# random coordinates
X = -20 * np.random.rand(100, 2)
X1 = 1 + 40 * np.random.rand(50, 2)
X[50:100, :] = X1
# draw dots
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
# K-Mean algorithm
Kmean = KMeans(n_clusters=3)
Kmean.fit(X)
C = Kmean.cluster_centers_
# draw cluster centers
plt.scatter(C[0][0], C[0][1], s=200, c='g', marker='s')
plt.scatter(C[1][0], C[1][1], s=200, c='r', marker='s')
plt.scatter(C[2][0], C[2][1], s=200, c='purple', marker='s')
plt.show()


# test
test_coords = np.array([3.0, 3.0]).reshape(-1, 1)
print(Kmean.predict(test_coords))
