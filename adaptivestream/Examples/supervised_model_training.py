import torch
import torch.nn as nn
import logging
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Wrapper.SupervisedModelWrapper import SupervisedModelWrapper
from sklearn.datasets import load_diabetes

"""
We present an example of how to train the SupervisedModelWrapper class

We will use the supervised regression task of predicting the diabetes disease progression for a patient, given a set of features.

Herein, we will follow the sample tutorial as outlined by Prasad at el. [1] 

[1] https://medium.com/analytics-vidhya/pytorch-for-deep-learning-feed-forward-neural-network-d24f5870c18
"""

class SimpleNet(nn.Module):
	def __init__(self, input_size, output_size):
		super(SimpleNet, self).__init__()
		self.l1 	= nn.Linear(input_size, 5)
		self.relu 	= nn.ReLU()
		self.l2 	= nn.Linear(5, output_size)
		return

	def forward(self, x):
		output = self.l1(x) 
		output = self.relu(output)
		output = self.l2(output)
		return output

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	data 	= load_diabetes(as_frame = True)
	feats   = data["data"]
	labels  = data["target"]

	# Ensure proper formatting of all input / output tensors
	feats_as_tensor 	= torch.tensor(feats.values).type(torch.float32)
	labels_as_tensor 	= torch.tensor(labels.values).reshape(-1,1).type(torch.float32)

	# Initialize and load data into the buffer
	buffer 	= LabelledFeatureBuffer()
	buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

	base_model 	= SimpleNet(feats_as_tensor.shape[1], 1)
	criterion	= torch.nn.MSELoss()
	optimizer 	= torch.optim.SGD(base_model.parameters(), lr = 0.001)

	model_wrapper = SupervisedModelWrapper(
						base_model 	= base_model,
						optimizer 	= optimizer,
						loss 		= criterion
					)

	model_wrapper.train(
		buffer 		= buffer,
		epoch  		= 1500,
		batch_size 	= 64, 
	)