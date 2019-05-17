# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:38:30 2019

@author: Robert
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import numpy as np
import shutil

CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

tf.executing_eagerly()

def decode_csv(row):
    columns = tf.decode_csv(row, record_defaults = DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    features.pop('key') # discard, not a real feature
    label = features.pop('fare_amount') # remove label from features and store
    return features, label

filenames_dataset = tf.data.Dataset.list_files(r'./*.csv', shuffle=False)
test = filenames_dataset._tensors

textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
item = textlines_dataset.make_one_shot_iterator().get_next()
print(item)

dataset = textlines_dataset.map(decode_csv)
item = dataset.make_one_shot_iterator().get_next()
