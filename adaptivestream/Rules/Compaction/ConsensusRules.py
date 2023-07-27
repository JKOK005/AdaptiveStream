import numpy as np
from Rules.Compaction.CompactionRules import CompactionRules

class ConsensusRules(CompactionRules):
    def __init__(self, variance):
        self.variance = variance

    def check_compaction(self, experts, buffer, *args, **kwargs):
        losses = []

        for model in experts:
            result = model.loss(buffer.get_data(), buffer.get_label())
            losses.append(result)
                
        if np.var(losses) < self.variance:
            return False
        else:
            return True
