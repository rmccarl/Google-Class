# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import google.cloud.bigquery as bq
import os

print(tf.__version__)

# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv(r'./taxi-train.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv(r'./taxi-valid.csv', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv(r'./taxi-test.csv', header = None, names = CSV_COLUMNS)

def make_train_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )
  

def make_eval_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )
  
def make_prediction_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )
  
def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns

tf.logging.set_verbosity(tf.logging.INFO)

OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols(), model_dir = OUTDIR)

model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10))


def print_rmse(model, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, df_valid)

predictions = model.predict(input_fn = make_prediction_input_fn(df_test))
for items in predictions:
  print(items)
  
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.DNNRegressor(hidden_units = [32, 8, 2],
      feature_columns = make_feature_cols(), model_dir = OUTDIR)
model.train(input_fn = make_train_input_fn(df_train, num_epochs = 100));
print_rmse(model, df_valid)

def create_query(phase, EVERY_N):
  """
  phase: 1 = train 2 = valid
  """
  base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  EXTRACT(DAYOFWEEK FROM pickup_datetime) * 1.0 AS dayofweek,
  EXTRACT(HOUR FROM pickup_datetime) * 1.0 AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
  CONCAT(CAST(pickup_datetime AS STRING), CAST(pickup_longitude AS STRING), CAST(pickup_latitude AS STRING), CAST(dropoff_latitude AS STRING), CAST(dropoff_longitude AS STRING)) AS key
FROM
  `nyc-tlc.yellow.trips`
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """

  if EVERY_N == None:
    if phase < 2:
      # Training
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), 4) < 2".format(base_query)
    else:
      # Validation
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), 4) = {1}".format(base_query, phase)
  else:
    query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), {1}) = {2}".format(base_query, EVERY_N, phase)
    
  return query

query1 = create_query(2, 100000)

def create_query2():
  base_query = """
SELECT 5
 """
  return base_query

query2 = create_query2()

client = bq.Client()

results = client.query(query2).result().to_dataframe()
results.head()

