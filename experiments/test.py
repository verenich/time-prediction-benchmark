import os
import pickle
from sys import argv
import numpy as np
import pandas as pd
import sys
import BucketFactory
from DatasetManager import DatasetManager

test_file = argv[1]
dataset_ref = argv[2]
bucket_encoding = "agg"
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]
optimal_params_filename = "training_params.json"

method_name = "%s_%s" % (bucket_method, cls_encoding)

home_dir = ""

# with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
#     best_params = pickle.load(fin)
best_params = pd.read_json(os.path.join(home_dir, optimal_params_filename), typ="series")

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

methods = encoding_dict[cls_encoding]

random_state = 22
fillna = True

##### MAIN PART ######

dataset_manager = DatasetManager(dataset_ref)
dtypes = {col: "object" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

for col in dataset_manager.label_col:
    dtypes[col] = "float"

test = pd.read_csv(test_file, sep=";", dtype=dtypes)
test[dataset_manager.timestamp_col] = pd.to_datetime(test[dataset_manager.timestamp_col])

# extract arguments
bucketer_args = {'encoding_method': bucket_encoding,
                 'case_id_col': dataset_manager.case_id_col,
                 'cat_cols': [dataset_manager.activity_col],
                 'num_cols': [],
                 'n_clusters': None,
                 'random_state': random_state}
if bucket_method == "cluster":
    bucketer_args['n_clusters'] = best_params[dataset_ref][method_name][cls_method]['n_clusters']

# Bucketing prefixes based on control flow
bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

with open('remtime_%s_%s_%s_%s.pkl' % (bucket_method, cls_encoding, cls_method, dataset_ref), 'rb') as f:
    pipelines = pickle.load(f)

# get predicted cluster for the test case
bucket_assignments_test = bucketer.predict(test)

# select relevant classifier
for bucket in set(bucket_assignments_test):
    if bucket not in pipelines:  # TODO fix this
        # use the general class ratio (in training set) as prediction
        sys.exit("No matching model has been trained!")

    else:
        # make actual predictions
        preds = pipelines[bucket].predict_proba(test)
        preds = max(0,np.rint(np.asscalar(preds)))
    print (preds)
