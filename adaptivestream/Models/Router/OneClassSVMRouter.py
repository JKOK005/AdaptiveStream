import tensorflow as tf
from Buffer.Buffer import Buffer
from sklearn import svm
from statistics import NormalDist
from Models.Router.Router import Router

class OneClassSVMRouter(Router):
	def __init__(self, 	init_params: dict, 
						*args, **kwargs
		):
		"""
		Parameters follow suite svm.OneClassSVM class in Scikit [1]

		[1]: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
		"""
		self.classifier = svm.OneClassSVM(**init_params)
		self.prob_dist  = None
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		feat 	= buffer.get_data()
		self.classifier.fit(feat)

		scores 	= self.classifier.score_samples(feat)
		self.prob_dist = NormalDist.from_samples(scores)
		return

	def permit_entry(self, 	input_X: tf.Tensor, 
					  		*args, **kwargs
			 		) -> bool:
		"""
		If class is 1 for input data, we permit entry.
		"""
		predicted_cls = self.classifier.predict(input_X)
		return predicted_cls[0] == 1

	def score(self, input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		score = self.classifier.score_samples(input_X)
		normalized_score = 1 / (score + 1e-3)
		return normalized_score.mean()

	def prob(self, 	input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		batch_score = self.score(input_X)
		return self.prob_dist.cdf(1 / batch_score)