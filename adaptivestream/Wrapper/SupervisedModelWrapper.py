import torch
import logging
from Buffer.Buffer import Buffer
from Wrapper.ModelWrapper import ModelWrapper

class SupervisedModelWrapper(torch.nn.Module, ModelWrapper):
	def __init__(self, 	base_model: torch.nn, 
						optimizer: torch.optim, 
						loss: torch.nn.modules.loss,
						*args, **kwargs
				):
		super(SupervisedModelWrapper, self).__init__()
		self.base_model = base_model
		self.optimizer 	= optimizer
		self.loss 		= loss
		self.logger  	= logging.getLogger("SupervisedModelWrapper")
		return

	def train(self, buffer: Buffer, 
					epoch: int, 
					batch_size: int,
					*args, **kwargs
			):

		buffer_feat 	= buffer.get_data()
		buffer_label 	= buffer.get_label()

		batched_feat 	= torch.tensor_split(buffer_feat, buffer_feat.shape[0] // batch_size, dim = 0) 		# Batch axis assumed to be the first
		batched_label   = torch.tensor_split(buffer_label, buffer_label.shape[0] // batch_size, dim = 0)

		for each_epoch in range(epoch):
			batch_cost 	= []

			for indx in range(len(batched_feat)):
				input_feat 	= batched_feat[indx]
				input_label = batched_label[indx]

				output 		= self.base_model(input_feat)
				cost 		= self.loss(output, input_label)
				batch_cost.append(cost)

				self.optimizer.zero_grad()
				cost.backward()
				self.optimizer.step()

			batch_cost_mean = torch.mean(torch.stack(batch_cost))
			batch_cost_mean_formatted = "{0:.4f}".format(batch_cost_mean)
			self.logger.info(f"Batch mean {batch_cost_mean_formatted}, Epoch {each_epoch}")
		return

	def infer(self, input_X, *args, **kwargs):
		pass