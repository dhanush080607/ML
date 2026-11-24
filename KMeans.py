import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

iris = load_iris()
X = pd.DataFrame(iris.data)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
gmm = GaussianMixture(n_components=3, random_state=42)

a = km.fit_predict(X)
b = gmm.fit_predict(X)

print("K-Means:", round(silhouette_score(X, a), 3))
print("EM GMM:", round(silhouette_score(X, b), 3))

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=a)
plt.show()