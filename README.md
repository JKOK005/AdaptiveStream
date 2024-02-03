# AdaptiveStream
`AdaptiveStream` is a framework harnessing capacity scaled Mixture of Expert models (MoE) for continual learning (CL) applications.

This repository guides the user on how to set up an `AdaptiveStream` pipeline and train our MoE ensemble on both the Airbnb & CORe50 dataset.

## Architecture
<img src="img/process_flow_adaptivestream.png" width=70% height=70%></img>

Continual learning models are constantly adapting towards distribution drifts within the data. 

Drift in real world data are a combination of sporadic and seasonal factors. Constant adjustments of a model's weights may induce the model to forget prior learnings which may be important for predicting seasonal patterns. 

`AdaptiveStream` offer end-to-end development of CL pipelines, deployable in $\approx$ 20 lines of code. 

The framework is primarily built on Tensorflow 2.11 and supported by multiple open sourced packages. 

### Buffers
In `AdaptiveStream`, incoming data is stored in an in-memory buffer. This buffer may be cleared after each scaling event. 

Current implentation supports the following buffer(s):

| Buffer type | Description |
| ------ | ------ |
| LabelledFeatureBuffer | Buffer to accommodate training features / labels for supervised learning tasks. |

More buffer classes may be implemented in the future.

### Scaling rules
`AdaptiveStream` uses Alibi-detect for concept drift detection. Other non-drift based rules are also offered.

Please see [scaling rules](adaptivestream/Rules/Scaling) for more details.

### Scaling policies
Unless a retention window is defined, all scaling events will clear the buffer. 

Please see [scaling policies](adaptivestream/Policies/Scaling) for more details.

### Compaction rules
Compaction helps limit the memory demads for capacity scaled MoE applications. This is especially important for long running applications where expert size may grow indefinitely.

Please see [compaction rules](adaptivestream/Rules/Compaction) for more details.

### Compaction policies
Compacting merges the last expert with the fallback expert. This shrinks the ensemble size by 1. 

Please see [compaction policies](adaptivestream/Policies/Compaction) for more details.

### MoE Ensemble
An ensemble consists of a collection of expert models. Each model is tuned on a specific snapshot of the data. We have 