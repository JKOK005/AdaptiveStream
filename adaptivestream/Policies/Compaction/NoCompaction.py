from Policies.Compaction import CompactionPolicy

class NoCompaction(CompactionPolicy):
	def compact(self, experts, 
					  fallback_expert,
					  *args, **kwargs
				):
		return experts, fallback_expert