import pandas as pd
import numpy as np
from numpy import array
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os
from sys import argv
from collections import defaultdict
import json
from tqdm import tqdm

import EncoderFactory
import BucketFactory
from DatasetManager import DatasetManager

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope

dataset_ref = argv[1]
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
n_iter = 3 #argv[5]
results_dir = "../results/CV/"

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

# dataset_ref = "bpic2015"
# bucket_encoding = "bool"
# bucket_method = "prefix"
# cls_encoding = "index"
# cls_method = "rf"
# results_dir = "/home/coderus/Temp/git/time-prediction-benchmark/results/"

method_name = "%s_%s"%(bucket_method, cls_encoding)

home_dir = ""

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011"],
    "bpic2015": ["bpic2015_%s"%municipality for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2017": ["bpic2017"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

# outfile = os.path.join(home_dir, results_dir, "cv_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref))

    
train_ratio = 0.8
n_splits = 3
random_state = 22


def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1

    start = time.time()
    score = 0
    for cv_iter in range(n_splits):

        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits):
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dataset_manager.case_id_col,
                         'cat_cols': [dataset_manager.activity_col],
                         'num_cols': [],
                         'random_state': random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = args["n_clusters"]
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        bucket_assignments_test = bucketer.predict(dt_test_prefixes)

        preds_all = []
        test_y_all = []
        if "prefix" in method_name:
            scores = defaultdict(int)
        for bucket in set(bucket_assignments_test):
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
            test_y = dataset_manager.get_label_numeric(dt_test_bucket)
            if len(relevant_train_cases_bucket) == 0:
                preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)
            else:
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                               relevant_train_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                if len(set(train_y)) < 2:
                    preds = [train_y[0]] * len(relevant_test_cases_bucket)
                else:
                    feature_combiner = FeatureUnion(
                        [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "rf":
                        cls = RandomForestRegressor(n_estimators=500,
                                                    max_features=args['max_features'],
                                                    max_depth=int(args['max_depth']),
                                                    random_state=random_state)

                    elif cls_method == "xgb":
                        cls = xgb.XGBRegressor(n_estimators=args['n_estimators'],
                                                # objective='binary:logistic',
                                                learning_rate=args['learning_rate'],
                                                subsample=args['subsample'],
                                                max_depth=int(args['max_depth']),
                                                colsample_bytree=args['colsample_bytree'],
                                                n_jobs=3,
                                                min_child_weight=int(args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "svm":
                        cls = SVR(C=2 ** args['C'],
                                  gamma=2 ** args['gamma'])

                    if cls_method == "svm":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
                    pipeline.fit(dt_train_bucket, train_y)

                    if cls_method == "svm":
                        preds = pipeline.predict(dt_test_bucket)
                    else:
                        preds = pipeline.predict(dt_test_bucket)

            if "prefix" in method_name:
                if len(test_y) < 2:
                    mae = None
                else:
                    mae = mean_absolute_error(test_y, preds)
                scores[bucket] += mae
            preds_all.extend(preds)
            test_y_all.extend(test_y)

        score += mean_absolute_error(test_y_all, preds_all)

    if "prefix" in method_name:
        for k, v in args.items():
            for bucket, bucket_score in scores.items():
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))
    else:
        for k, v in args.items():
            fout_all.write(
                "%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))
    pbar.update()
    fout_all.flush()
    return {'loss': score / n_splits, 'status': STATUS_OK, 'model': cls}


##### MAIN PART ######    
# with open(outfile, 'w') as fout:
#
#     fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "cls", ";".join(bucketer_params_names), ";".join(cls_params_names), "nr_events", "metric", "score"))
    
for dataset_name in datasets:

    dataset_manager = DatasetManager(dataset_name)

    # read the data
    data = dataset_manager.read_dataset()

    # split data into train and test
    train, _ = dataset_manager.split_data(data, train_ratio)

    # consider prefix lengths until 90% of positive cases have finished
    min_prefix_length = 1
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    # del data

    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'fillna': True}

    # prepare chunks for CV
    dt_prefixes = []
    class_ratios = []
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        class_ratios.append(array([np.mean(train_chunk["remtime"])]))
        # generate data where each prefix is a separate instance
        dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
    del train
    print(dt_prefixes[0].shape)

    # set up search space
    if cls_method == "rf":
        space = {
            'max_features': hp.uniform('max_features', 0, 1),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1))
        }
    elif cls_method == "xgb":
        space = {
            'n_estimators': scope.int(hp.choice('n_estimators', np.arange(100, 800, step=1))),
            'learning_rate': hp.uniform("learning_rate", 0, 0.3),
            'subsample': hp.uniform("subsample", 0.5, 1),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
            'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))
        }
    elif cls_method == "svm":
        space = {
            'C': hp.uniform('C', -15, 15),
            'gamma': hp.uniform('gamma', -15, 15)
        }
    if bucket_method == "cluster":
        space['n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 8, 1))

    # optimize parameters
    trial_nr = 1
    trials = Trials()
    fout_all = open(os.path.join(home_dir, results_dir, "param_optim_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    if "prefix" in method_name:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))
    else:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))
    pbar = tqdm(total=n_iter, desc="Hyperopt")
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
    pbar.close()
    fout_all.close()

    # write the best parameters
    best_params = hyperopt.space_eval(space, best)
    # outfile = os.path.join(home_dir, results_dir, "Optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    # write to file
    # with open(outfile, "wb") as fout:
    #     pickle.dump(best_params, fout)
    #
    # with open("%s.json" % outfile, "w") as fout:
    #     json.dump(best_params, fout, indent=3)
