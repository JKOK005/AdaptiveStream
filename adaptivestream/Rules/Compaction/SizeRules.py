from Rules.Compaction.CompactionRules import CompactionRules

class SizeRules(CompactionRules):
    def __init__(self, N, K):
        self.N = N
        self.K = K

    def check_compaction(self, experts, *args, **kwargs):
        if len(experts) < self.N + self.K:
            return False
        else:
            return True
