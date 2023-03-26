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
		self.training_params 	= training_params
		self.inference_params 	= inference_params
		return

	def train(self, buffer: Buffer,
					*args, **kwargs
			):
		feat = buffer.get_data()
		self.classifier.fit(feat, **self.training_params)
		return

	def permit_entry(self, 	input_X, 
					  		*args, **kwargs
	  		) -> bool:
		self.inference_params["outlier_type"] = "instance"
		pred 	= self.classifier.predict(input_X, **self.inference_params)
		return pred["data"]["is_outlier"][0] == 0