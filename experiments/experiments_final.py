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

import EncoderFactory
import BucketFactory
from DatasetManager import DatasetManager

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

dataset_ref = argv[1]
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
gap = 1 # int(argv[5])
n_iter = 3 # int(argv[6])
results_dir = "../results/"

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

# dataset_ref = "bpic2015"
# bucket_encoding = "bool"
# bucket_method = "single"
# cls_encoding = "agg"
# cls_method = "rf"

method_name = "%s_%s"%(bucket_method, cls_encoding)

home_dir = ""

# create results directory
if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))


dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011"],
    "bpic2015": ["bpic2015%s"%municipality for municipality in range(1,6)],
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


train_ratio = 0.8
random_state = 22

    
##### MAIN PART ######    

for dataset_name in datasets:

    # load optimal params
    optimal_params_filename = os.path.join(home_dir, "optimal_params", "%s_%s_%s.pkl" % (cls_method, dataset_name, method_name))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue

    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)

    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    # split data into train and test
    train, test = dataset_manager.split_data(data, train_ratio)

    # consider prefix lengths until 90% of positive cases have finished
    min_prefix_length = 1
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    del data

    outfile = os.path.join(home_dir, results_dir, "Final_results_%s_%s_%s.csv" % (cls_method, method_name, dataset_name))

    start_test_prefix_generation = time.time()
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    test_prefix_generation_time = time.time() - start_test_prefix_generation
    print(dt_test_prefixes.shape)

    cls_encoder_args = {'case_id_col':dataset_manager.case_id_col,
                        'static_cat_cols':dataset_manager.static_cat_cols,
                        'static_num_cols':dataset_manager.static_num_cols,
                        'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols':dataset_manager.dynamic_num_cols,
                        'fillna':True}

    offline_total_times = []
    online_event_times = []
    train_prefix_generation_times = []
    for ii in range(n_iter):
        print(f"Starting iteration {ii}")
        # create prefix logs
        start_train_prefix_generation = time.time()
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
        train_prefix_generation_time = time.time() - start_train_prefix_generation
        train_prefix_generation_times.append(train_prefix_generation_time)

        # extract arguments
        bucketer_args = {'encoding_method':bucket_encoding,
                         'case_id_col':dataset_manager.case_id_col,
                         'cat_cols':[dataset_manager.activity_col],
                         'num_cols':[],
                         'random_state':random_state}
        if bucket_method == "cluster":
            bucketer_args['n_clusters'] = int(args["n_clusters"])

        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        start_offline_time_bucket = time.time()
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        offline_time_bucket = time.time() - start_offline_time_bucket

        bucket_assignments_test = bucketer.predict(dt_test_prefixes)

        preds_all = []
        test_y_all = []
        nr_events_all = []
        offline_time_fit = 0
        current_online_event_times = []
        for bucket in set(bucket_assignments_test):
            if bucket_method == "prefix":
                current_args = args[bucket]
            else:
                current_args = args
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)

            nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
            if len(relevant_train_cases_bucket) == 0:
                preds = array([np.mean(train["remtime"])] * len(relevant_test_cases_bucket))
                current_online_event_times.extend([0] * len(preds))
            else:
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                               relevant_train_cases_bucket)  # one row per event
                train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                if len(set(train_y)) < 2:
                    preds = [train_y[0]] * len(relevant_test_cases_bucket)
                    current_online_event_times.extend([0] * len(preds))
                    test_y_all.extend(dataset_manager.get_label_numeric(dt_test_bucket))
                else:
                    start_offline_time_fit = time.time()
                    feature_combiner = FeatureUnion(
                        [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "rf":
                        cls = RandomForestRegressor(n_estimators=500,
                                                     max_features=current_args['max_features'],
                                                     random_state=random_state)

                    elif cls_method == "xgb":
                        cls = xgb.XGBRegressor(n_estimators=int(current_args['n_estimators']),
                                            #objective='binary:logistic',
                                            learning_rate=current_args['learning_rate'],
                                            subsample=current_args['subsample'],
                                            max_depth=int(current_args['max_depth']),
                                            colsample_bytree=current_args['colsample_bytree'],
                                            n_jobs=3,
                                            #min_child_weight=int(current_args['min_child_weight']),
                                            seed=random_state)

                    elif cls_method == "svm":
                        cls = SVR(C=2 ** current_args['C'],
                              gamma=2 ** current_args['gamma'])

                    if cls_method == "svm":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                    pipeline.fit(dt_train_bucket, train_y)
                    offline_time_fit += time.time() - start_offline_time_fit

                    # predict separately for each prefix case
                    preds = []
                    test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                    for _, group in test_all_grouped:

                        test_y_all.extend(dataset_manager.get_label_numeric(group))

                        start = time.time()
                        _ = bucketer.predict(group)
                        if cls_method == "svm":
                            pred = pipeline.predict(group)
                        else:
                            pred = pipeline.predict(group)

                        pipeline_pred_time = time.time() - start
                        current_online_event_times.append(pipeline_pred_time / len(group))
                        pred = pred.clip(min=0)  # if remaining time is predicted to be negative, make it zero
                        preds.extend(pred)

            preds_all.extend(preds)

        offline_total_time = offline_time_bucket + offline_time_fit + train_prefix_generation_time
        offline_total_times.append(offline_total_time)
        online_event_times.append(current_online_event_times)


    with open(outfile, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score", "nr_cases"))

        # fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "test_prefix_generation_time", test_prefix_generation_time, -1))

        # for ii in range(len(offline_total_times)):
        #     fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "train_prefix_generation_time", train_prefix_generation_times[ii], -1))
        #     fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "offline_time_total", offline_total_times[ii], -1))
        #     fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "online_time_avg", np.mean(online_event_times[ii]), -1))
        #     fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "online_time_std", np.std(online_event_times[ii]), -1))

        dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
        for nr_events, group in dt_results.groupby("nr_events"):
            if len(group.actual) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "mae", np.nan, len(group.actual)))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "mae", mean_absolute_error(group.actual, group.predicted), len(group.actual)))
        # fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, "avg_mae", mean_absolute_error(dt_results.actual, dt_results.predicted), len(dt_results.actual)))

        # online_event_times_flat = [t for iter_online_event_times in online_event_times for t in iter_online_event_times]
        # fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat), -1))
        # fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat), -1))
        # fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_total_times), -1))
        # fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_total_times), -1))
