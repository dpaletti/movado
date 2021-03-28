import numpy as np
from sklearn.neighbors import KDTree, DistanceMetric
a = np.array([1, 2, 3])
b = np.empty([0])
b = np.append(b, 2)
print(type(np.mean(b)))
print(KDTree.valid_metrics)
