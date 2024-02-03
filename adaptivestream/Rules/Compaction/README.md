Current implementation supports the following kinds of compaction rules(s):

| Rule | Description |
| ------ | ------ |
| SizeRules | When expert capacity exceeds a threshold. |
| TimeRules | Triggered when the system has been running for a defined time period. |
| ConsensusRules | When the average variance of the ensemble has exceeded a threshold variance for the current dataset. | 