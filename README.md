# AdaptiveStream
Transforms offline, batch based model training process into an online process

## Scaling experts

### Drift detection rules to implement for scaling decision
For tabular data (all algorithms from alibi-detect)
1) Online Maximum Mean Discrepancy 
2) Least-Squares Density Difference
3) Online Cramer-von mises
4) Online Fishers Exact Test

### Outlier detection algorithms to implement for expert gates
For tabular data (all algorithms from alibi-detect unless specified)
1) Outlier variable autoencoder
2) Isolation forest
3) OneClassSVM (scikit)
4) Variational auto-encoding gaussian mixture model