import tensorflow as tf
from Buffer.Buffer import Buffer
from sklearn import svm
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
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		feat = buffer.get_data()
		self.classifier.fit(feat)
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
		return 1 / (score[0] + 1e-3)