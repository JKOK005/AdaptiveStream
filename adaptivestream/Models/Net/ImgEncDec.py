import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout

class ImgEncDec(object):
	@classmethod
	def get_encoder(cls, input_shape: tf.Tensor, output_size: int):
		pass 

	@classmethod
	def get_decoder(cls, input_size, output_shape: tf.Tensor):
		pass