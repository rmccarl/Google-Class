# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:12:19 2019

@author: Robert
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()



import numpy as np
import shutil

def do_it():
    
    print(tf.__version__)
    
    # In CSV, label is the first column, after the features, followed by the key
    CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
    DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]
    
    def read_dataset(filename, mode, batch_size = 512):
        
     
      def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults = DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        features.pop('key') # discard, not a real feature
        label = features.pop('fare_amount') # remove label from features and store
        return features, label
    
      # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(filename, shuffle=False)
      print(filenames_dataset._tensors)
      
      # Read lines from text files
      textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
      
      # Parse text lines as comma-separated values (CSV)
      dataset = textlines_dataset.map(decode_csv)
    
      # Note:
      # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
      # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
    
      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = None # loop indefinitely
          dataset = dataset.shuffle(buffer_size = 10 * batch_size, seed=2)
      else:
          num_epochs = 1 # end-of-input after this
    
      dataset = dataset.repeat(num_epochs).batch(batch_size)
    
      return dataset
    
    def get_train_input_fn():
      return read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN)
    
    def get_valid_input_fn():
      return read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL)
    
    INPUT_COLUMNS = [
        tf.feature_column.numeric_column('pickuplon'),
        tf.feature_column.numeric_column('pickuplat'),
        tf.feature_column.numeric_column('dropofflat'),
        tf.feature_column.numeric_column('dropofflon'),
        tf.feature_column.numeric_column('passengers'),
    ]
    
    def add_more_features(feats):
      # Nothing to add (yet!)
      return feats
    
    feature_cols = add_more_features(INPUT_COLUMNS)
    
    tf.logging.set_verbosity(tf.logging.INFO)
    OUTDIR = 'taxi_trained'
    shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
    
    tf.executing_eagerly()
    
    input_fn = get_train_input_fn()
    
    model = tf.estimator.LinearRegressor(
          feature_columns = feature_cols, model_dir = OUTDIR)
    model.train(input_fn = input_fn, steps = 200)
    
    metrics = model.evaluate(input_fn = get_valid_input_fn, steps = None)
    print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
    
    return


def main():

  do_it()
  return


if __name__ == "__main__":
    main()

