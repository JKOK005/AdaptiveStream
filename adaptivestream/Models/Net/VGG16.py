import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout

class VggNet16Factory(object):
	@classmethod
	def get_model(cls, input_shape):
		return 	Sequential([
					Input(shape = input_shape),
					Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (2, 2), strides = 2),
					Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 128, kernel_size = (3,3), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (2, 2), strides = 2),
					Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 256, kernel_size = (3,3), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (2, 2), strides = 2),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (2, 2), strides = 2),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					Conv2D(filters = 512, kernel_size = (3,3), padding = "same", activation = "relu"),
					MaxPooling2D(pool_size = (2, 2), strides = 2),
					Flatten(),
					Dense(units = 4096, activation="relu"),
					Dropout(0.5),
					Dense(units = 4096, activation="relu"),
					Dropout(0.5),
					Dense(units = 5, activation="softmax"),
				])