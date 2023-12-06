import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout, Lambda

class CaffeNetFactory(object):
	@classmethod
	def get_model(cls, input_shape: (int), output_size: int):
		return 	Sequential([
					Input(shape = input_shape),
					Conv2D(filters = 96, kernel_size = (11, 11), strides=(4, 4), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
					Lambda(tf.nn.local_response_normalization),
					Conv2D(filters = 256, kernel_size = (5,5), strides=(1, 1), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
					Lambda(tf.nn.local_response_normalization),
					Conv2D(filters = 384, kernel_size = (3,3), strides=(1, 1), padding = "same", activation = "relu"),
					Conv2D(filters = 384, kernel_size = (3,3), strides=(1, 1), padding = "same", activation = "relu"),
					Conv2D(filters = 256, kernel_size = (3,3), strides=(1, 1), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
					Flatten(),
					Dense(units = 4096, activation="relu"),
					Dropout(0.5),
					Dense(units = 4096, activation="relu"),
					Dropout(0.5),
					Dense(units = 4096, activation="relu"),
					Dropout(0.5),
					Dense(units = output_size, activation="softmax"),
				])