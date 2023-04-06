import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re
from sklearn.metrics import mean_squared_error
import math

# from loglizer.logparser import Drain
from LogPai.loglizer import preprocessing

#====================================================================================================================================================
def get_train_test_data(dataset: pd.DataFrame, train_date_start, train_date_end, test_date_start, test_date_end):
  
  train_data = dataset[train_date_start : train_date_end]

  test_data = dataset[test_date_start : test_date_end]

  return train_data, test_data

def get_X_Y(dataset: pd.DataFrame, y_column):
  #Drop rows with NaN values
  XY = dataset.dropna()
  #Split data to X and Y
  X = XY.drop(y_column, axis=1)
  Y = XY[y_column]

  return X,Y

def normalize_X_Y_train(X: pd.DataFrame, Y: pd.DataFrame):
  scaler_input = MinMaxScaler(feature_range=(0, 1))
  scaler_input = scaler_input.fit(X)

  scaler_target = MinMaxScaler(feature_range=(0, 1))
  scaler_target = scaler_target.fit(Y)

  X_normalized = pd.DataFrame(scaler_input.transform(X), columns=X.columns, index=X.index)
  Y_normalized = pd.DataFrame(scaler_target.transform(Y), columns=Y.columns, index= Y.index)

  return X_normalized, Y_normalized, scaler_input, scaler_target

def normalize_X_Y_test(X: pd.DataFrame, Y: pd.DataFrame, scaler_input, scaler_target):
  X_normalized = pd.DataFrame(scaler_input.transform(X), columns=X.columns, index=X.index)
  Y_normalized = pd.DataFrame(scaler_target.transform(Y), columns=Y.columns, index= Y.index)

  return X_normalized, Y_normalized

def get_trained_reg(x: pd.DataFrame, y: pd.DataFrame):
  return LinearRegression().fit(x, y)

def get_trained_MLP(x: pd.DataFrame, y: pd.DataFrame, model=None, dense_1_shape=10, dense_2_shape=10, dense_3_shape=10, dense_4_shape=10, epochs=100, batch_size=100, verbose=2):
  #Train model further
  if model:
    model.fit(x, y, epochs=epochs, batch_size = batch_size, verbose = verbose, shuffle=False)
    return model

  #Otherwise, create new model
  model = Sequential([
            Input(x.shape[1]),
            Dense(dense_1_shape, activation='relu'),
            #sigmoid
            #regularization (L2)
            Dense(dense_2_shape, activation='relu'),
            Dense(y.shape[1])
        ])
  model.summary()

  #Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')

  #Train model
  callback = EarlyStopping(monitor='loss', patience=20, mode='min', restore_best_weights=True)
  model.fit(x, y, epochs=epochs, batch_size = batch_size, verbose = verbose, shuffle=False, callbacks=[callback])

  return model

def plot_predicted_vs_true(predictions: pd.DataFrame, ground_truth: pd.DataFrame, y_column, hue=None, x_column='index', figsize=(25, 7)):

  #predictions = predictions.rename(columns={y_column : y_column + '_pred'})
  #ground_truth = ground_truth.rename(columns={y_column : y_column + '_true'})
  fig, ax = plt.subplots(figsize=figsize)
  if hue:
    ax.plot(getattr(ground_truth[ground_truth[hue] == False], x_column), ground_truth[ground_truth[hue] == False][y_column], label='Measured signal (not {})'.format(hue), color='blue', marker='o')
    ax.scatter(getattr(ground_truth[ground_truth[hue] == True], x_column), ground_truth[ground_truth[hue] == True][y_column], label='Measured signal ({})'.format(hue), color='darkred', marker='x', s=100)
  else:
    ax.plot(getattr(ground_truth, x_column), ground_truth[y_column], label='Measured signal', color='blue', marker='o')

  ax.plot(getattr(predictions, x_column), predictions[y_column], label='Simulation', color='green')

  ax.grid(True)

  ax.legend()
  ax.set_xlabel('Timestamp')
  ax.set_ylabel( y_column + ' [0C]')

  '''
  df = predictions.join(ground_truth)

  fig = plt.figure(figsize=(25,7))
  sns.lineplot(data=df)
  '''
  '''
  fig, ax = plt.subplots(figsize=(25,7))
  ax.plot(ground_truth.index, ground_truth[y_column], color = 'green')
  ax.plot(predictions.index, predictions[y_column], color = 'blue')

  ax.grid(True)

  ax.legend(['Ground Truth', 'Simulation'])
  ax.set_xlabel('Timestamp')
  ax.set_ylabel(y_column + ' [0C]')
  '''

def plot_error(predictions: pd.DataFrame, ground_truth: pd.DataFrame, y_column):
  fig, ax = plt.subplots(figsize=(25,7))
  errors = ((ground_truth[y_column] - predictions[y_column] ) / ground_truth[y_column]) * 100
  ax.plot(errors, color = 'red')

  ax.grid(True)

  ax.set_xlabel('Timestamp')
  ax.set_ylabel('Error [%]')

  return pd.DataFrame(errors)

def implement_pipeline_reg(dataset: pd.DataFrame, train_date_start, train_date_end, test_date_start, test_date_end, y_column, turbine_id=None, features=None):
  if turbine_id:
    dataset = dataset[dataset.Turbine_ID == turbine_id]
  if features:
    dataset = dataset[features]

  #Split dataset to training and test
  train_data, test_data = get_train_test_data(dataset, train_date_start, train_date_end, test_date_start, test_date_end)
  
  #X-Y split
  X_train, Y_train = get_X_Y(train_data, y_column)
  X_test, Y_test = get_X_Y(test_data, y_column)
  
  #Normalize data
  #x_train, y_train, scaler_input, scaler_target = normalize_X_Y_train(X_train, Y_train)
  
  #Train model
  print('Training model...')
  model = get_trained_reg(X_train, Y_train)

  #Calculate errors
  predictions_train = pd.DataFrame(model.predict(X_train), columns=Y_train.columns, index=X_train.index)
  errors_train = pd.DataFrame(Y_train[y_column] - predictions_train[y_column])

  predictions_test = pd.DataFrame(model.predict(X_test), columns=Y_test.columns, index=X_test.index)
  errors_test = pd.DataFrame(Y_test[y_column] - predictions_test[y_column])

  #Evaluate model
  #x_test, y_test = normalize_X_Y_test(X_test, Y_test, scaler_input, scaler_target)
  print('Evaluating model...')
  # print('Score for training data: ', model.score(X_train, Y_train))
  # print('Score for test data: ', model.score(X_test, Y_test))
  print('RMSE Train: ', math.sqrt(mean_squared_error(Y_train[y_column], predictions_train[y_column])))
  print('RMSE Test: ', math.sqrt(mean_squared_error(Y_test[y_column], predictions_test[y_column])))

  return dataset, X_train, Y_train, predictions_train, errors_train, X_test, Y_test, predictions_test, errors_test, model


def implement_pipeline_mlp(dataset: pd.DataFrame, train_date_start, train_date_end, test_date_start, test_date_end, y_column, turbine_id=None, features=None,
                           model=None, dense_1_shape=32, dense_9_shape=16, epochs=10, batch_size=100, verbose=2, normalize=False):
  if turbine_id:
    dataset = dataset[dataset.Turbine_ID == turbine_id]
  if features:
    dataset = dataset[features]

  #Split dataset to training and test
  train_data, test_data = get_train_test_data(dataset, train_date_start, train_date_end, test_date_start, test_date_end)
  
  #X-Y split
  X_train, Y_train = get_X_Y(train_data, y_column)
  X_test, Y_test = get_X_Y(test_data, y_column)
  
  if normalize:
    #Normalize data
    x_train, y_train, scaler_input, scaler_target = normalize_X_Y_train(X_train, Y_train)
    
    #Train model
    print('Training model...')
    model = get_trained_MLP(x_train, y_train, epochs=epochs)

    x_test, y_test = normalize_X_Y_test(X_test, Y_test, scaler_input, scaler_target)

    #Calculate errors
    predictions_train = pd.DataFrame(scaler_target.inverse_transform(model.predict(x_train)), columns=Y_train.columns, index=x_train.index)
    errors_train = pd.DataFrame(Y_train[y_column] - predictions_train[y_column])
    rmse_train = math.sqrt(mean_squared_error(Y_train[y_column], predictions_train[y_column]))

    predictions_test = pd.DataFrame(scaler_target.inverse_transform(model.predict(x_test)), columns=Y_test.columns, index=x_test.index)
    errors_test = pd.DataFrame(Y_test[y_column] - predictions_test[y_column])
    rmse_test = math.sqrt(mean_squared_error(Y_test[y_column], predictions_test[y_column]))

    #Evaluate model
    #x_test, y_test = normalize_X_Y_test(X_test, Y_test, scaler_input, scaler_target)
    print('Evaluating model...')
    # print('Score for training data: ', model.score(X_train, Y_train))
    # print('Score for test data: ', model.score(X_test, Y_test))
    print('RMSE Train: ', rmse_train)
    print('RMSE Test: ', rmse_test)

    return dataset, X_train, Y_train, predictions_train, errors_train, X_test, Y_test, predictions_test, errors_test, model, scaler_input, scaler_target, rmse_train, rmse_test


  else:
    #Train model
    print('Training model...')
    model = get_trained_MLP(X_train, Y_train, epochs=epochs)

    #Evaluate model
    print('Evaluating model...')
    print('Score for training data: ', model.evaluate(X_train, Y_train))
    print('Score for test data: ', model.evaluate(X_test, Y_test))

    #Calculate errors
    predictions_train = pd.DataFrame(model.predict(X_train), columns=Y_train.columns, index=X_train.index)
    #predictions_train = pd.DataFrame(scaler_target.inverse_transform(model.predict(x_train)), columns=Y_train.columns, index=x_train.index)
    errors_train = pd.DataFrame(Y_train[y_column] - predictions_train[y_column])

    predictions_test = pd.DataFrame(model.predict(X_test), columns=Y_test.columns, index=X_test.index)
    #predictions_test = pd.DataFrame(scaler_target.inverse_transform(model.predict(x_test)), columns=Y_test.columns, index=x_test.index)
    errors_test = pd.DataFrame(Y_test[y_column] - predictions_test[y_column])

    return dataset, X_train, Y_train, predictions_train, errors_train, X_test, Y_test, predictions_test, errors_test, model

  
def remove_duplicates(list_with_duplicates):
    return list(dict.fromkeys(list_with_duplicates))

def kl_divergence(p, q):
  epsilon = 1e-10
  p += epsilon
  q += epsilon
  return np.sum(p * np.log(p / q))

def get_log_feature(logs: pd.DataFrame, log_msg_column, string_query):
  filtered_logs = logs[logs[log_msg_column].str.contains(string_query).fillna(False)]
  index = filtered_logs.index.name
  filtered_logs = filtered_logs.reset_index().drop_duplicates().set_index(index)

  log_feature = filtered_logs[log_msg_column].str.findall(string_query + ' *(\d)*').apply(lambda row: int(row[0]))

  return log_feature

def get_log_features(logs: pd.DataFrame, log_msg_column, string_query):
  feats = logs[logs[log_msg_column].str.contains(string_query, case=False, na=False)][log_msg_column]
  feats = pd.DataFrame(feats.apply(lambda row: row.split(',')[0]))
  feats = feats[log_msg_column].str.rsplit('.', 1, expand=True).rename(columns = {0 : 'name', 1 : 'value'})
  feats.value = feats.value.astype(int)
  feats = feats.pivot(columns='name', values='value')
  return list(feats.columns), feats

def get_target_columns(all_columns, string_query_list):
  import pandas as pd

  filtered_columns = pd.Series(all_columns)
  for string_query in string_query_list:
    filtered_columns = filtered_columns[filtered_columns.str.contains(string_query)]
  return list(filtered_columns)

def append_relevant_log_warnings(Y: pd.DataFrame, logs: pd.DataFrame, log_msg_column, target_feature_name, filter_by=None, max_distance='24H'):
  relevant_logs = logs[logs[log_msg_column].str.contains(target_feature_name.split('_')[0], na=False, case=False)]

  Y['logs_found'] = None
  for index, row in Y[Y[filter_by]].iterrows() if filter_by else Y.iterrows():
    filtered_logs = relevant_logs[(index - pd.Timedelta(max_distance)) : index]
    if not filtered_logs.empty:
      string = ''
      for i, r in filtered_logs.iterrows():
        string = string + r[log_msg_column] + ' @ ' + str(i) + '\n '
      Y.loc[index, 'logs_found'] = string

    else:
      Y.loc[index, 'logs_found'] = 'No logs found'
    
  return Y

def plot_log_warnings(gt: pd.DataFrame, anomaly_column_name, log_column_name, target_feature, replacement_message = ' '):
  last_logs_found = ''
  for line in range(0, gt.shape[0], 1):
    if gt[anomaly_column_name][line] and gt[log_column_name][line] != last_logs_found:
      plt.text(gt.index[line], gt[target_feature][line], gt[log_column_name][line], size='small', color='black', weight='semibold', rotation=30)
      last_logs_found = gt[log_column_name][line]

    elif gt[log_column_name][line] == replacement_message:
        plt.text(gt.index[line], gt[target_feature][line], '!!{}!!'.format(replacement_message), size='medium', color='darkred', weight='bold', rotation=30)

def plot_control_signals(x: pd.DataFrame, log_feature_name):
  last_value = -1
  for ix, row in x.iterrows():
    if str(row[log_feature_name]) == 'nan':
      plt.axvline(x=ix, linestyle='-', color='grey', label='Day Separator')
      last_value = -1
    if row[log_feature_name] != last_value:
      if row[log_feature_name] == 0.0:
        plt.axvline(x=ix, linestyle='--', color='darkred', label='{} 0'.format(log_feature_name))
      elif row[log_feature_name] == 1:
        plt.axvline(x=ix, linestyle='--', color='green', label='{} 1'.format(log_feature_name))
      elif row[log_feature_name] == 2:
        plt.axvline(x=ix, linestyle='--', color='black', label='{} 2'.format(log_feature_name))
      
      last_value=row[log_feature_name]


def load_data(data_folder, file_name_1, file_name_2=None, time_index_column=None, delimeter=','):
  if file_name_2:
      all_data = pd.concat(
          [
          pd.read_csv(data_folder + file_name_1, delimiter=delimeter), 
          pd.read_csv(data_folder + file_name_2, delimiter=delimeter)
          ], 
          ignore_index=True
          )
  else:
      all_data = pd.read_csv(data_folder + file_name_1, delimiter=delimeter)
  
  if time_index_column:
      all_data[time_index_column] = pd.to_datetime(all_data[time_index_column], utc=True)
      all_data.set_index(time_index_column, inplace=True)
      all_data = all_data.sort_index()
  
  return all_data


def get_log_pai_feats(UnitTile, target_log_ft, data_folder_input, data_folder_output, log_file_name, log_format, depth=4, st=0.5, regex=[]):
    #Parse logs
    # parser = Drain.LogParser(log_format=log_format, indir=data_folder_input, outdir=data_folder_output,  depth=depth, st=st, rex=regex)
    # parser.parse(log_file_name)

    #Read parsed logs
    df_log_parsed_all = load_data(
        data_folder=data_folder_output, 
        file_name_1=log_file_name+'_structured.csv',
        time_index_column='TimeDetected'
    )
    df_log_parsed = df_log_parsed_all[df_log_parsed_all.UnitTitle == UnitTile]
    #df_log_struct_all = pd.read_csv(data_folder_output+log_file_name+'_templates.csv')

    #Extract log features
    feature_extractor = preprocessing.FeatureExtractor()
    x_train_logs = feature_extractor.fit_transform(df_log_parsed[target_log_ft], term_weighting='tf-idf', 
                                                normalization='zero-mean')

    logpai_feats = pd.DataFrame(
        data=x_train_logs, index=df_log_parsed.index, 
        columns=['logpai_ft_{}_{}'.format(target_log_ft, i) for i in range(1, x_train_logs.shape[1] + 1)]
        )

    return logpai_feats

def append_alarm_info(df, n=10, q=0.0):
    #Add difference in daytime (in days)
    df['Diff_in_days'] = df.index.day.to_series().diff().values

    '''
    Append number of Anomalies for each data point
    At the start of the day (Diff_in_days > 0) start the counter of anomalies for the days
    '''
    df['n_Anomaly'] = 0
    for index, row in df[df['Diff_in_days'] != 0].sort_index().iterrows():
        df.loc[index, 'n_Anomaly'] = 1 if row['Anomaly'] else 0
    
    '''
    For all other data points (not start of the day), add 1 to the previous anomaly counter 
    if current data point was labeled as anomaly
    '''
    for index, row in df[df['Diff_in_days'] == 0].iterrows():
        i = df.index.get_loc(index)
        df['n_Anomaly'].iloc[i] = df.iloc[i-1]['n_Anomaly'] + 1 if row['Anomaly'] else df.iloc[i-1]['n_Anomaly']
    
    #Send alarm after n detected anomalies in a given day

    #Handle special types of threshold if n not given directly as raw number
    if not isinstance(n, int):
        anomalies = df[df['Anomaly']]
        if n == 'mean + 3std':
            n = np.floor(anomalies['n_Anomaly'].mean() + 3 * anomalies['n_Anomaly'].std())
        
        elif n == 'max - min':
            n = np.floor(anomalies['n_Anomaly'].max() - anomalies['n_Anomaly'].min())
        
        elif n == 'quantile':
            n = np.floor(anomalies['n_Anomaly'].quantile(q))


    df['Alarm'] = (df['n_Anomaly'] >= n) & df['Anomaly']
    return n, df