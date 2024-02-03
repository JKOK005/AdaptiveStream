Current implementation supports the following kinds of scaling policies(s):

| Rule | Description |
| ------ | ------ |
| ExpertOnlyScaling | Scaling expert model template without router. |
| NaiveScaling | Scaling expert & router with random weight initialization. | 
| NaiveKnowledgeTransfer | Scaling expert & router with fine tuning from previous expert. |

More classes may be implemented in the future.
