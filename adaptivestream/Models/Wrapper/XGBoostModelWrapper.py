import logging
from Buffer.Buffer import Buffer
from Models.Wrapper.ModelWrapper import ModelWrapper
from xgboost import XGBModel

class XGBoostModelWrapper(ModelWrapper):
	def __init__(self, 	xg_boost_model: XGBModel,
						init_params: dict,
						training_params: dict,
						*args, **kwargs
				):
		super(XGBoostModelWrapper, self).__init__()
		self.xg_boost_model 	= xg_boost_model(**init_params)
		self.training_params 	= training_params
		self.logger  			= logging.getLogger("XGBoostModelWrapper")
		return

	def train(	self, buffer: Buffer, 
				*args, **kwargs
			):
		buffer_feat 	= buffer.get_data()
		buffer_label 	= buffer.get_label()

		self.xg_boost_model.fit(buffer_feat, buffer_label, **self.training_params)
		return

	def infer(	self, input_X: tf.Tensor, 
					  predict_params: dict,
					  *args, **kwargs
			):
		return self.xg_boost_model.predict(input_X, **predict_params)