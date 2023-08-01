from Rules.Compaction.CompactionRules import CompactionRules

class TimeRules(CompactionRules):
    def __init__(self, time_limit):
        self.time_limit = time_limit

    def check_compaction(self, buffer, *args, **kwargs):
        [min_ts, max_ts] = buffer.get_batch_timestamps()
        interval = max_ts - min_ts
        if interval < self.time_limit:
            return False
        else:
            return True
