class Expert(object):
	self.base_model = base_model
	self.router = router

	def __init__(self, 	base_model, 
						router
				):
		pass

	def permit_entry(self, input_data):
		return self.router.is_within_distribution(input_data)

	def infer(self, input_data):
		pass