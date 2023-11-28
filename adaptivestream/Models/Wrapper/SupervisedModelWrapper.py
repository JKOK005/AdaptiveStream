import copy
import tensorflow as tf
import logging
import random
from Buffer.Buffer import Buffer
from Models.Wrapper.ModelWrapper import ModelWrapper

def load_balance_gpu(fn):
	gpus = tf.config.list_physical_devices('GPU')

	def wrapper(*args, **kwargs):
		selected_gpu = random.randint(0, len(gpus) -1)

		with tf.device(f"/device:GPU:{selected_gpu}"):
			print(f"Using GPU: {selected_gpu}")
			kwargs["GPU_NUM"] = selected_gpu
			return fn(*args, **kwargs)

	return wrapper

class SupervisedModelWrapper(ModelWrapper):
	def __init__(self, 	base_model: tf.keras.Model, 
						optimizer: tf.keras.optimizers, 
						loss_fn: tf.keras.losses,
						training_params: dict,
						training_batch_size: int,
						*args, **kwargs
				):
		super(SupervisedModelWrapper, self).__init__()
		self.loss_fn 				= loss_fn
		self.model 					= base_model
		self.optimizer 				= optimizer
		self.training_params 		= training_params
		self.training_batch_size 	= training_batch_size
		self.logger  				= logging.getLogger("SupervisedModelWrapper")
		self.GPU_NUM 				= 0
		return

	@load_balance_gpu
	def __deepcopy__(self, memo, *args, **kwargs):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result

		# Deep copy all other attributes
		for k, v in self.__dict__.items():
			if k == "base_model":
				setattr(result, k, tf.keras.models.clone_model(v))

			elif k == "optimizer" or k == "loss_fn":
				setattr(result, k, v)

			elif k == "GPU_NUM":
				setattr(result, k, kwargs["GPU_NUM"])

			else:
				setattr(result, k, copy.deepcopy(v, memo))

			print(k, v)

		# Return updated instance
		return result

	def train(	self, buffer: Buffer, 
				*args, **kwargs
			):
		buffer_feat 	= buffer.get_data()
		buffer_label 	= buffer.get_label()
		dataset 		= tf.data.Dataset.from_tensor_slices((buffer_feat, buffer_label)) \
										 .batch(self.training_batch_size)

		self.model.compile(optimizer = self.optimizer, loss = self.loss_fn)
		self.model.fit(x = dataset, **self.training_params)
		return

	def infer(	self, input_X: tf.Tensor, 
				*args, **kwargs
			):
		return self.model(input_X)

	def loss(self, 	input_X: tf.Tensor,
					ground_truth: tf.Tensor,
					*args, **kwargs
			) -> tf.Tensor:
		pred = self.infer(input_X = input_X)
		return self.loss_fn(ground_truth, pred)