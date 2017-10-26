import os
import pickle
from sys import argv

import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion

import BucketFactory
import ClassifierFactory
import EncoderFactory
from DatasetManager import DatasetManager

train_file = argv[1]
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
n_min_cases_in_bucket = 30

##### MAIN PART ######    

dataset_manager = DatasetManager(dataset_ref)
dtypes = {col: "object" for col in dataset_manager.dynamic_cat_cols + dataset_manager.static_cat_cols +
          [dataset_manager.case_id_col, dataset_manager.timestamp_col]}
for col in dataset_manager.dynamic_num_cols + dataset_manager.static_num_cols:
    dtypes[col] = "float"

for col in dataset_manager.label_col:
    dtypes[col] = "float"

train = pd.read_csv(train_file, sep=";", dtype=dtypes)
train[dataset_manager.timestamp_col] = pd.to_datetime(train[dataset_manager.timestamp_col])
train = train.sort_values(dataset_manager.timestamp_col, ascending=True, kind='mergesort')

# consider prefix lengths until 90th percentile of case length
min_prefix_length = 1
# max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
max_prefix_length = min(10, dataset_manager.get_pos_case_length_quantile(train, 0.90))

# create prefix logs
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)

print(dt_train_prefixes.shape)

# extract arguments
bucketer_args = {'encoding_method': bucket_encoding,
                 'case_id_col': dataset_manager.case_id_col,
                 'cat_cols': [dataset_manager.activity_col],
                 'num_cols': [],
                 'n_clusters': None,
                 'random_state': random_state}
if bucket_method == "cluster":
    bucketer_args['n_clusters'] = best_params[dataset_ref][method_name][cls_method]['n_clusters']

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': fillna}

# Bucketing prefixes based on control flow
print("Bucketing prefixes...")
bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

pipelines = {}

# train and fit pipeline for each bucket
for bucket in set(bucket_assignments_train):
    print("Fitting pipeline for bucket %s..." % bucket)

    # set optimal params for this bucket
    if bucket_method == "prefix":
        cls_args = {k: v for k, v in best_params[dataset_ref][method_name][cls_method][u'%s'%bucket].items() if
                    k not in ['n_clusters', 'n_neighbors']}
    else:
        cls_args = {k: v for k, v in best_params[dataset_ref][method_name][cls_method].items() if
                    k not in ['n_clusters', 'n_neighbors']}
    cls_args['random_state'] = random_state
    cls_args['min_cases_for_training'] = n_min_cases_in_bucket

    # select relevant cases
    relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                   relevant_cases_bucket)  # one row per event
    train_y = dataset_manager.get_label_numeric(dt_train_bucket)

    feature_combiner = FeatureUnion(
        [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
    pipelines[bucket] = Pipeline(
        [('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])

    pipelines[bucket].fit(dt_train_bucket, train_y)

with open('remtime_%s_%s_%s_%s.pkl' % (bucket_method, cls_encoding, cls_method, dataset_ref), 'wb') as f:
    pickle.dump(pipelines, f, protocol=2)
