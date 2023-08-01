import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from Buffer.LabelledFeatureBuffer import LabelledFeatureBuffer
from Rules.Compaction import TimeRules
from sklearn.datasets import load_diabetes


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    red = pd.read_csv(
        "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-red.csv", sep=';'
    )
    red = np.asarray(red, np.float32)
    
    white = pd.read_csv(
        "https://storage.googleapis.com/seldon-datasets/wine_quality/winequality-white.csv", sep=';'
    )
    white = np.asarray(white, np.float32)

    white_feats_as_tensor = tf.convert_to_tensor(white, dtype = tf.float32)
    white_labels_as_tensor = tf.ones([white_feats_as_tensor.shape[0], 1])
    red_feats_as_tensor = tf.convert_to_tensor(red, dtype = tf.float32)
    red_labels_as_tensor = tf.ones([red_feats_as_tensor.shape[0], 1])
    training_feats = white_feats_as_tensor[:2048]
    training_labels = white_labels_as_tensor[:2048]

    buffer = LabelledFeatureBuffer()
    time_rules = TimeRules(time_limit=10)
    
    for t in range(0, 100):
        buffer.add(batch_input = (training_feats, training_labels), batch_timestamp=t)
        flag = time_rules.check_compaction(buffer)
        if flag:
            print(f"Need to perform compaction in {t} timesteps")
            buffer.clear()
        else:
            print(f"No need to perform compaction in {t} timesteps")
