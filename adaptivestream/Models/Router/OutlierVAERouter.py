import tensorflow as tf
from alibi_detect.od import OutlierVAE
from Buffer.Buffer import Buffer
from Models.Router.Router import Router

class OutlierVAERouter(Router):
	def __init__(self, 	init_params: dict,
						training_params: dict,
						inference_params: dict,
						*args, **kwargs
				):
		"""
		Parameters follow suite alibi_detect.od.OutlierVAE class in alibi_detect
		Please use alibi_detect Tensorflow backend

		Ref: https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vae.html
		"""
		self.classifier 		= OutlierVAE(**init_params)
		self.prob_dist  		= None
		self.training_params 	= training_params
		self.inference_params 	= inference_params
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		feat 	= buffer.get_data()
		self.classifier.fit(feat, **self.training_params)

		self.inference_params["outlier_type"] 			= "instance"
		self.inference_params["return_instance_score"] 	= True
		pred 	= self.classifier.predict(feat, **self.inference_params)
		scores 	= spred["data"]["instance_score"]
		self.prob_dist = NormalDist.from_samples(1 / (scores + 1e-3))
		return

	def permit_entry(self, 	input_X: tf.Tensor, 
					  		*args, **kwargs
	  				) -> bool:
		"""
		Performs outlier detection on a single input_x target.
		If is_outlier result is 0, the data is accepted and passes through the gate.
		"""
		self.inference_params["outlier_type"] 			= "instance"
		pred = self.classifier.predict(input_X, **self.inference_params)
		return pred["data"]["is_outlier"][0] == 3

	def score(self, input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		self.inference_params["outlier_type"] 			= "instance"
		self.inference_params["return_instance_score"] 	= True
		pred = self.classifier.predict(input_X, **self.inference_params)
		return pred["data"]["instance_score"].mean()

	def prob(self, 	input_X: tf.Tensor,
					*args, **kwargs
			) -> float:
		batch_score = self.score(input_X)
		return self.prob_dist.cdf(1 / batch_score)