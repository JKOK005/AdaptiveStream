import datetime
import tensorflow as tf
from Buffer.LabelledBuffer import LabelledBuffer

class LabelledFeatureBuffer(LabelledBuffer):
	feat 			= None
	label 			= None
	
	feat_latest 	= None
	label_latest 	= None

	feat_window  	= []
	label_window  	= []
	window_size  	= 7 	# TODO: Make this variable
	last_cleared 	= 0

	def __init__(self, *args, **kwargs):
		super(LabelledFeatureBuffer, self).__init__(*args, **kwargs)
		return

	def get_batch_timestamps(self) -> [int, int]:
		return self.batch_timestamp_range

	def get_data(self) -> tf.Tensor:
		return tf.concat(self.feat_window + [self.feat], axis = 0)

	def get_data_latest(self) -> tf.Tensor:
		return self.feat_latest

	def get_label(self) -> tf.Tensor:
		return tf.concat(self.label_window + [self.label], axis = 0)

	def get_label_latest(self) -> tf.Tensor:
		return self.label_latest

	def get_count(self) -> int:
		return self.feat.shape[0]

	def get_last_cleared(self) -> datetime.datetime:
		return self.last_cleared

	def add(self, 	batch_input: (tf.Tensor),
					batch_timestamp: int = -1,
					*args, **kwargs
			):
		"""
		Input data is expected to be a tuple of (feat, labels)
		"""
		input_feat 	= batch_input[0]
		input_label = batch_input[1]

		if self.batch_timestamp_range[0] == -1:
			self.batch_timestamp_range[0] = batch_timestamp
		else:
			self.batch_timestamp_range[1] = batch_timestamp

		if self.feat is not None:
			self.feat = tf.concat([self.feat, input_feat], axis = 0)
		else:
			self.feat = input_feat

		if self.label is not None:
			self.label = tf.concat([self.label, input_label], axis = 0)
		else:
			self.label = input_label

		self.feat_latest 	= input_feat
		self.label_latest 	= input_label
		return

	def clear(self):
		self.feat_window.append(self.feat)
		self.feat_window  	= self.feat_window[-1 * self.window_size : ]

		self.label_window.append(self.label)
		self.label_window  	= self.label_window[-1 * self.window_size : ]

		self.feat 			= None
		self.label 			= None

		self.feat_latest 	= None
		self.label_latest 	= None

		self.last_cleared 	= datetime.datetime.now()
		self.batch_timestamp_range 	= [-1, -1]
		return
