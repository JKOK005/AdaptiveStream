import torch
from Buffer.Buffer import Buffer
from Wrapper.ModelWrapper import ModelWrapper

class SupervisedModelWrapper(ModelWrapper, torch.nn.Module):
	def __init__(self, 	base_model: torch.nn, 
						optimizer: torch.optim, 
						loss: torch.nn.modules.loss,
						*args, **kwargs
				):
		self.optimizer 	= optimizer
		self.loss 		= loss
		super(SupervisedModelWrapper).__init__(base_model = base_model, *args, **kwargs)
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
			for indx in range(len(batched_feat)):
				self.optimizer.zero_grad()

				input_feat 	= batched_feat[indx]
				input_label = batched_label[indx]
				
				output 		= self.base_model(input_feat)
				loss 		= self.loss(output, input_label)

				self.loss.backward()
				self.optimizer.step()
		return
