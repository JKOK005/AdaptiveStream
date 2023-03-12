from abc import ABC

class StreamWrapper(ABC):
	def __init__(self, 	base_model,
						router,
						scaling_strategy,
						compaction_strategy,
						ingestion_source,
				):
		"""
		: param base_model 			: Foundational model for each expert
		: param router 				: Router class used to decide expert allocation during inference
		: param scaling_strategy 	: ScalingStrategy class used to scale experts
		: param compaction_strategy : CompactionStrategy class used to perform compaction
		: param ingestion_source 	: Source class used to fetch data for training
		"""
		self.base_model 			= base_model
		self.router 				= router
		self.scaling_strategy  		= scaling_strategy
		self.compaction_strategy 	= compaction_strategy
		self.ingestion_source 		= ingestion_source
		return

	@abstractmethod
	def _validate(self):
		"""
		Inheriting class to implement validation logic to ensure compatibility among all components of the StreamWrapper class
		"""
		pass

	def _run(self):
		"""
		Continuously streams data from ingestion_source and updates the model
		"""
		pass

	def start(self):
		pass