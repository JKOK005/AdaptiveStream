import numpy as np
from sklearn.cluster import KMeans

class ClusterTools(object):
	@staticmethod
	def k_means_cluster(indexes: np.ndarray, 
						clusters: int, 
						**params
					) -> np.ndarray:
		"""
		Calls sklearn kmeans clustering module to cluster data points

		Additional params can be found in [1]

		[1]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
		"""
		kmeans = KMeans(n_clusters = clusters, **params).fit(indexes)
		return kmeans.labels_

	@staticmethod
	def exemplar_selection(indexes: np.ndarray) -> int:
		"""
		Given a r x n `indexes` matrice, where r represents the number of experts and n the dimension of the index,
		return the expert whose index is closest to the cluster mean. 
		"""
		mean_of_index = np.mean(indexes, axis = 0)
		dist_to_mean  = np,sum((indexes - mean_of_index) ** 2, axis = 1)
		return int(np.argmin(dist_to_mean, axis = 0))