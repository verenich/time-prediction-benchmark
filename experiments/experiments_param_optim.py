import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
from numpy import array
import os
from sys import argv
import itertools

import EncoderFactory
import BucketFactory
import ClassifierFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
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

# bucketing params to optimize 
if bucket_method == "cluster":
    bucketer_params = {'n_clusters':[2, 4, 6]}
else:
    bucketer_params = {'n_clusters':[1]}

# classification params to optimize
if cls_method == "rf":
    cls_params = {'n_estimators':[250, 500],
                  'max_features':["sqrt", 0.1, 0.5, 0.75]}

    
elif cls_method == "xgb":
    cls_params = {'n_estimators': [100,500],
    'learning_rate': [0.02,0.04,0.06],
    'subsample': [0.5,0.8],
    'max_depth': [3,5,7],
    'colsample_bytree': [0.6,0.9]}

bucketer_params_names = list(bucketer_params.keys())
cls_params_names = list(cls_params.keys())


outfile = os.path.join(home_dir, results_dir, "cv_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref))

    
train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "cls", ";".join(bucketer_params_names), ";".join(cls_params_names), "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        
        # split data into train and test
        train, _ = dataset_manager.split_data(data, train_ratio)
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        del data
        
        part = 0
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=3):
            part += 1
            print("Starting chunk %s..."%part)
            sys.stdout.flush()
            
            # create prefix logs
            dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)
            
            print(dt_train_prefixes.shape)
            print(dt_test_prefixes.shape)
            
            
            for bucketer_params_combo in itertools.product(*(bucketer_params.values())):
                for cls_params_combo in itertools.product(*(cls_params.values())):
                    print("Bucketer params are: %s"%str(bucketer_params_combo))
                    print("Cls params are: %s"%str(cls_params_combo))

                    # extract arguments
                    bucketer_args = {'encoding_method':bucket_encoding, 
                                     'case_id_col':dataset_manager.case_id_col, 
                                     'cat_cols':[dataset_manager.activity_col], 
                                     'num_cols':[], 
                                     'random_state':random_state}
                    for i in range(len(bucketer_params_names)):
                        bucketer_args[bucketer_params_names[i]] = bucketer_params_combo[i]

                    cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                                        'static_cat_cols':dataset_manager.static_cat_cols,
                                        'static_num_cols':dataset_manager.static_num_cols, 
                                        'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                                        'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                                        'fillna':fillna}

                    cls_args = {'random_state':random_state,
                                'min_cases_for_training':n_min_cases_in_bucket}
                    for i in range(len(cls_params_names)):
                        cls_args[cls_params_names[i]] = cls_params_combo[i]
        
                                   
                    # Bucketing prefixes based on control flow
                    print("Bucketing prefixes...")
                    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
                    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

                    pipelines = {}

                    # train and fit pipeline for each bucket
                    for bucket in set(bucket_assignments_train):
                        print("Fitting pipeline for bucket %s..."%bucket)
                        relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
                        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
                        train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                        pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])
                        pipelines[bucket].fit(dt_train_bucket, train_y)

                        
                    # if the bucketing is prefix-length-based, then evaluate for each prefix length separately, otherwise evaluate all prefixes together 
                    max_evaluation_prefix_length = max_prefix_length if bucket_method == "prefix" else min_prefix_length
                    
                    prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()

                    for nr_events in range(min_prefix_length, max_evaluation_prefix_length+1):
                        print("Predicting for %s events..."%nr_events)

                        if bucket_method == "prefix":
                            # select only prefixes that are of length nr_events
                            relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

                            if len(relevant_cases_nr_events) == 0:
                                break

                            dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
                            del relevant_cases_nr_events
                        else:
                            # evaluate on all prefixes
                            dt_test_nr_events = dt_test_prefixes.copy()

                        start = time()
                        # get predicted cluster for each test case
                        bucket_assignments_test = bucketer.predict(dt_test_nr_events)

                        # use appropriate classifier for each bucket of test cases
                        # for evaluation, collect predictions from different buckets together
                        preds = []
                        test_y = []
                        for bucket in set(bucket_assignments_test):
                            relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
                            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events, relevant_cases_bucket) # one row per event

                            if len(relevant_cases_bucket) == 0:
                                continue

                            elif bucket not in pipelines:
                                # use mean remaining time (in training set) as prediction
                                preds_bucket = array([np.mean(train_chunk["remtime"])] * len(relevant_cases_bucket))
                                # preds_bucket = [dataset_manager.get_class_ratio(train_chunk)] * len(relevant_cases_bucket)

                            else:
                                # make actual predictions
                                preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

                            preds_bucket = preds_bucket.clip(min=0)  # if remaining time is predicted to be negative, make it zero
                            preds.extend(preds_bucket)

                            # extract actual label values
                            test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                            test_y.extend(test_y_bucket)

                        if len(test_y) < 2:
                            mae = None
                        else:
                            mae = mean_absolute_error(test_y, preds)
                        #prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")
                        bucketer_params_str = ";".join([str(param) for param in bucketer_params_combo])
                        cls_params_str = ";".join([str(param) for param in cls_params_combo])
                                   
                        fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "mae", mae))
                        #fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "precision", prec))
                        #fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "recall", rec))
                        #fout.write("%s;%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, bucketer_params_str, cls_params_str, nr_events, "fscore", fscore))

                    print("\n")
