import tensorflow as tf
from alibi_detect.od import OutlierVAE
from Buffer.Buffer import Buffer
from Models.Router import Router

class OutlierVAERouter(object):
	def __init__(self, 	encoder_net: tf.keras.Model, 
						decoder_net: tf.keras.Model,
						*args, **kwargs
				):
		"""
		Parameters follow suite alibi_detect.od.OutlierVAE class in alibi_detect
		Please use alibi_detect Tensorflow backend

		Ref: https://docs.seldon.io/projects/alibi-detect/en/stable/od/methods/vae.html
		"""
		self.classifier = OutlierVAE(**kwargs)
		return

