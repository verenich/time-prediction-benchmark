# Based on https://github.com/verenich/ProcessSequencePrediction

import time
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import csv
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys
from DatasetManager_LSTM import DatasetManager
from sys import argv
import glob

import pandas as pd
import numpy as np


dataset_ref = argv[1]
lstmsize = int(argv[2])
dropout = float(argv[3])
n_layers = int(argv[4])
batch_size = int(argv[5])
learning_rate = float(argv[6])
activation = argv[7]
optimizer = argv[8]
nb_epoch = 500

train_ratio = 0.8
val_ratio = 0.2
cls_method = "lstm"

home_dir = ""
results_dir = "../results/"
detailed_results_dir = "../results/detailed/"
checkpoint_dir = "../results/chkpnt_%s" % dataset_ref

checkpoint_prefix = os.path.join(home_dir, checkpoint_dir, "model_%s_%s_%s_%s_%s_%s_%s"%(lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate))
checkpoint_filepath = "%s.{epoch:02d}-{val_loss:.2f}.hdf5" % checkpoint_prefix

model_filename = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
print(model_filename)
outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv" % (
    cls_method, dataset_ref, lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate))
detailed_results_file = os.path.join(home_dir, detailed_results_dir, "detailed_%s_%s_%s_%s_%s_%s_%s_%s_%s.csv" % (
    dataset_ref, cls_method, lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate))

method_name = "%s_%s_%s_%s_%s_%s_%s"%(lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate)

##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_ref)
data = dataset_manager.read_dataset()

train, test = dataset_manager.split_data(data, train_ratio=train_ratio)
train, val = dataset_manager.split_data(train, train_ratio=1 - val_ratio)

dt_train = dataset_manager.encode_data(train)
dt_test = dataset_manager.encode_data(test)


max_prefix_length = min(20, dataset_manager.get_case_length_quantile(data, 0.90))

activity_cols = [col for col in dt_train.columns if col.startswith("act")]
n_activities = len(activity_cols)
data_dim = dt_train.shape[1] - 3

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

main_input = Input(shape=(max_prefix_length, data_dim), name='main_input')

if n_layers == 1:
    l2_3 = LSTM(lstmsize, input_shape=(max_prefix_length, data_dim), implementation=2,
                kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(main_input)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 2:
    l1 = LSTM(lstmsize, activation=activation, input_shape=(max_prefix_length, data_dim), implementation=2,
              kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(
        main_input)  # the shared layer
    b1 = BatchNormalization(axis=1)(l1)
    l2_3 = LSTM(lstmsize, activation=activation, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b1)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 3:
    l1 = LSTM(lstmsize, input_shape=(max_prefix_length, data_dim), implementation=2,
              kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(
        main_input)  # the shared layer
    b1 = BatchNormalization(axis=1)(l1)
    l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(b1) # the shared layer
    b2 = BatchNormalization(axis=1)(l2)
    l3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b2)
    b2_3 = BatchNormalization()(l3)

outcome_output = Dense(1, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output')(b2_3)

model = Model(inputs=[main_input], outputs=[outcome_output])
if optimizer == "adam":
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
elif optimizer == "rmsprop":
    opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'outcome_output':'mean_absolute_error'}, optimizer=opt)
model.load_weights(model_filename)

print('Evaluating...')

detailed_results = pd.DataFrame()
with open(outfile, 'w') as fout:
    csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["dataset", "method", "cls", "nr_events", "metric", "score", "nr_cases"])

    total = 0
    total_mae = 0
    total_rmse = 0
    for nr_events in range(1, max_prefix_length + 1):
        # encode only prefixes of this length
        X, y_o, case_ids = dataset_manager.generate_LSTM_data_prefix_length(dt_test, max_prefix_length, nr_events)
        print(X.shape, y_o.shape)
        if X.shape[0] == 0:
            break

        pred_y_o = model.predict(X, verbose=0)
        mae = mean_absolute_error(y_o[:, 0], pred_y_o[:, 0])
        rmse = np.sqrt(mean_squared_error(y_o[:, 0], pred_y_o[:, 0]))

        print("prefix = %s, n_cases = %s, mae = %s" % (nr_events, X.shape[0], mae))
        total += X.shape[0]
        total_mae += mae * X.shape[0]
        total_rmse += rmse * X.shape[0]

        # csv_writer.writerow([dataset_ref, cls_method, nr_events, "n_cases", X.shape[0]])
        csv_writer.writerow([dataset_ref, method_name, cls_method, nr_events, "mae", mae, X.shape[0]])
        csv_writer.writerow([dataset_ref, method_name, cls_method, nr_events, "rmse", rmse, X.shape[0]])

        # current_results = pd.DataFrame(
        #     {"dataset": dataset_ref, "cls": cls_method, "nr_events": nr_events,
        #      "predicted": pred_y_o[:, 0], "actual": y_o[:, 0], "case_id": case_ids})
        # detailed_results = pd.concat([detailed_results, current_results])

    csv_writer.writerow([dataset_ref, method_name, cls_method, -1, "total_mae", total_mae / total, total])
    csv_writer.writerow([dataset_ref, method_name, cls_method, -1, "total_rmse", total_rmse / total, total])

print("total mae: ", total_mae / total)

#detailed_results.to_csv(detailed_results_file, sep=";", index=False)
