import datetime
import torch
from Buffer.LabelledBuffer import LabelledBuffer

class LabelledFeatureBuffer(LabelledBuffer):
	feat 			= None
	label 			= None
	count   		= 0
	last_cleared 	= 0

	def get_data(self):
		return self.feat

	def get_label(self):
		return self.label

	def get_count(self):
		return self.count

	def get_last_cleared(self):
		return self.last_cleared

	def add(self, 	batch_input: (torch.Tensor), 
					*args, **kwargs
			):
		"""
		Input data is expected to be a tuple of (feat, labels)
		"""
		input_feat 	= batch_input[0]
		input_label = batch_input[1]

		if self.feat is not None:
			self.feat = torch.vstack(self.feat, input_feat)
		else:
			self.feat = input_feat

		if self.label is not None:
			self.label = torch.vstack(self.label, input_label)
		else:
			self.label = input_label

		self.count += self.feat.shape[0]
		return

	def clear(self):
		self.feat 			= None
		self.label 			= None
		self.count 			= 0
		self.last_cleared 	= datetime.datetime.now()
