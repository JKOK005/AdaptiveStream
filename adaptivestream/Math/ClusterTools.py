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
	def exemplar_selection(	indexes: np.ndarray, 
							exemplar_count: int
					) -> [int]:
		"""
		Given a r x n `indexes` matrice, where r represents the number of experts and n the dimension of the index,
		return K = exemplar_count experts that represent the cluster.

		1) The expert whose index is closest to the cluster mean will always be part of the exemplar set.
		2) The remaining K - 1 experts will be sampled from the remaining set of experts. 
		"""
		mean_of_index = np.mean(indexes, axis = 0)
		dist_to_mean  = np.sum((indexes - mean_of_index) ** 2, axis = 1)
	
		return_index  = []
		return_index.append(dist_to_mean.argmin()) 		# In ascending order of dist
		
		cum_dist 	  = dist_to_mean * 0
		for i in range(exemplar_count -1):
			prev_exemplar_index = indexes[return_index[-1]]
			cum_dist  += np.sum((indexes - prev_exemplar_index) ** 2, axis = 1)
			cum_index = cum_dist.argsort()

			for i in reversed(cum_index):
				if i not in return_index:
					return_index.append(i)
					break
		return return_index
