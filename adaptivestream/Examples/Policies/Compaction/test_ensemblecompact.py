import logging
import tensorflow as tf
from alibi_detect.models.tensorflow.losses import elbo
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Models.Router.OutlierVAERouter import OutlierVAERouter
from Models.Wrapper.SupervisedModelWrapper import SupervisedModelWrapper
from Policies.Scaling.NaiveScaling import NaiveScaling
from sklearn.datasets import load_diabetes
from tensorflow.keras import layers, losses, optimizers, Sequential
from Rules.Compaction import SizeRules
from Policies.Compaction import EnsembleCompaction
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def build_router(input_size):
    encoder_net = Sequential([
        tf.keras.Input(shape = (input_size, )),
        layers.Dense(10, activation = "relu"),
        layers.Dense(5, activation = "relu"),
        ]
    )
    
    decoder_net = Sequential([
        tf.keras.Input(shape = (16, )),
        layers.Dense(5, activation = "relu"),
        layers.Dense(10, activation = "relu"),
        layers.Dense(input_size),
        ]
    )
    
    return OutlierVAERouter(
        init_params = {
            "threshold" : 0.75,
            "score_type" : "mse",
            "encoder_net" : encoder_net,
            "decoder_net" : decoder_net,
            "latent_dim" : 16,
            "samples" : 10},
        
        training_params = {
            "loss_fn" : elbo,
            "optimizer" : optimizers.legacy.Adam(learning_rate=1e-3),
            },
        
        inference_params = {
            "outlier_perc" 	: 80
            }
        )

def build_model_wrapper(input_size, output_size):
    base_model = Sequential(
        [
            tf.keras.Input(shape = (input_size, )),
            layers.Dense(5, activation = "relu"),
            layers.Dense(output_size, activation = "relu")
        ]
    )
            
    criterion = losses.MeanSquaredError()
    optimizer = optimizers.legacy.SGD(learning_rate = 0.0001)
    return SupervisedModelWrapper(
        base_model = base_model,
        optimizer = optimizer,
        loss_fn = criterion,
        training_params = {
            "epochs" : 10,
            "batch_size" : 64
            }
    )


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    data = load_diabetes(as_frame = True)
    feats = data["data"]
    labels = data["target"]

    # Ensure proper formatting of all input / output tensors
    feats_as_tensor = tf.convert_to_tensor(feats.values, dtype = tf.float32)[:-2]
    labels_as_tensor = tf.convert_to_tensor(labels.values, dtype = tf.float32)[:-2]
    labels_as_tensor = tf.reshape(labels_as_tensor, [len(labels_as_tensor), 1])

    # Initialize and load data into the buffer
    buffer = LabelledFeatureBuffer()
    buffer.add(batch_input = (feats_as_tensor, labels_as_tensor))

    experts = []
    k = 2
    n = 3
    strategies = 'merge'
    size_rules = SizeRules(N=n, K=k)
    ensemble_compact = EnsembleCompaction(N=n, K=k, strategy=strategies)

    fallback_expert = None

    for i in range(15):
        model_wrapper = build_model_wrapper(input_size = feats_as_tensor.shape[1], output_size = 1)
        router = build_router(input_size = feats_as_tensor.shape[1])
        policy = NaiveScaling(model = model_wrapper, router = router)
        policy.set_buffer(buffer = buffer)
        new_expert = policy.train_expert()

        experts.append(new_expert)
        flag = size_rules.check_compaction(experts)
        if flag:
            print(f"Need to perform compaction in {i}-th model")
            fallback_expert, experts = ensemble_compact.compact(experts, fallback_expert, buffer)
            time.sleep(5)
        else:
            print(f"No need to perform compaction in {i}-th model")
            time.sleep(5)
