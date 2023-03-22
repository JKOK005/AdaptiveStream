from Buffer.Buffer import Buffer
from sklearn import svm
from Models.Router import Router

class OneClassSVMRouter(Router):
	def __init__(self, *args, **kwargs):
		"""
		Parameters follow suite svm.OneClassSVM class in Scikit

		Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
		"""
		self.classifier = svm.OneClassSVM(**kwargs)
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		feat = buffer.get_data()
		self.classifier.fit(feat)
		return

	def permit_entry(self, 	input_X, 
					  		*args, **kwargs
			 		):
		"""
		If class is 1 for input data, we permit entry.
		"""
		predicted_cls = self.classifier.predict(input_X)
		return predicted_cls[0] == 1