from Policies.Compaction import CompactionPolicy

class NoCompaction(CompactionPolicy):
	def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
		return fallback_expert, experts