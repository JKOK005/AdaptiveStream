from Policies.Compaction import CompactionPolicy

class LastRemovalCompaction(CompactionPolicy):
	def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
		return experts[0], experts[1:]