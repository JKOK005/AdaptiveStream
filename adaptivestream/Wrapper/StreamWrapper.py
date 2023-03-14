from abc import ABC
from multiprocessing import Process

class StreamWrapper(ABC):
	def __init__(self, 	ensemble,
						ingestion_source,
				):
		"""
		: param ensemble 			: ExpertEnsemble class
		: param ingestion_source 	: Source class used to fetch data for training
		"""
		self.ensemble 			= ensemble
		self.ingestion_source 	= ingestion_source
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
		stream_handler = Process(target = self._run)
		stream_handler.start()
		return stream_handler