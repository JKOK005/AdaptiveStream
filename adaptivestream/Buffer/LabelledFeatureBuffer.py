import datetime
import tensorflow as tf
from Buffer.LabelledBuffer import LabelledBuffer

class LabelledFeatureBuffer(LabelledBuffer):
	feat 			= None
	feat_latest 	= None
	label 			= None
	label_latest 	= None
	count   		= 0
	last_cleared 	= 0

	def __init__(self, *args, **kwargs):
		super(LabelledFeatureBuffer, self).__init__(*args, **kwargs)
		return

	def get_data(self) -> tf.Tensor:
		return self.feat

	def get_data_latest(self) -> tf.Tensor:
		return self.feat_latest

	def get_label(self) -> tf.Tensor:
		return self.label

	def get_label_latest(self) -> tf.Tensor:
		return self.label_latest

	def get_count(self) -> int:
		return self.count

	def get_last_cleared(self) -> datetime.datetime:
		return self.last_cleared

	def add(self, 	batch_input: (tf.Tensor), 
					*args, **kwargs
			):
		"""
		Input data is expected to be a tuple of (feat, labels)
		"""
		input_feat 	= batch_input[0]
		input_label = batch_input[1]

		if self.feat is not None:
			self.feat = tf.concat([self.feat, input_feat], axis = 0)
		else:
			self.feat = input_feat

		if self.label is not None:
			self.label = tf.concat([self.label, input_label], axis = 0)
		else:
			self.label = input_label

		self.count 			+= self.feat.shape[0]
		self.feat_latest 	= input_feat
		self.label_latest 	= input_label
		return

	def clear(self):
		self.feat 			= None
		self.feat_latest 	= None
		self.label 			= None
		self.label_latest 	= None
		self.count 			= 0
		self.last_cleared 	= datetime.datetime.now()
		return
