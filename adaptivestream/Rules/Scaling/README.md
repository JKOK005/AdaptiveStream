Current implementation supports the following kinds of scaling rules(s):

| Rule | Description |
| ------ | ------ |
| BufferSizeLimit | Scaling when the buffer size hits a hard threshold. |
| MaxDuration | Scales when data has been accummulated beyond a time period in the buffer since last clearance. |
| TimeLimit | (For timestamped data) Scales when the data has exceeded a time period, as defined by the `batch_timestamp` property during ingestion | 
| OnlineMMDDrift | Online MMD Drift [detector](https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/onlinemmddrift.html) | 

More classes may be implemented in the future.
