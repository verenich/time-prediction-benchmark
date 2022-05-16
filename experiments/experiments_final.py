import pandas as pd
import numpy as np
from numpy import array
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
import json
import os
from sys import argv


import EncoderFactory
import BucketFactory
import ClassifierFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
bucket_method = argv[2]
cls_encoding = argv[3]
cls_method = argv[4]
optimal_params_filename = "training_params.json"
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

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
    best_params = json.load(fin)
    #best_params = pickle.load(fin)
print(best_params)
dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011"],
    "bpic2015": ["bpic2015%s"%municipality for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2017": ["bpic2017"]
}

encoding_dict = {
    "last": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}

#print(best_params["traffic_fines"][method_name][cls_method].items())
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref)) 

    
train_ratio = 0.8
random_state = 22
fillna = True
n_min_cases_in_bucket = 30


#function to calculate relative error
def get_rae(x,y, y_min):
    # print(x,y,y_min)
    e = abs(x-y)
    d = max(y_min,y)
    return e/d



    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score", "nr_cases"))
    
    for dataset_name in datasets:

        #print(best_params[dataset_name][method_name][cls_method].items())
        
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        datacopy = data

        # # remove incomplete traces i.e. those ending with send fine
        # segment_indices = pd.read_csv("logdata/incomplete_cases.csv")["Case ID"]
        # indexes = set(segment_indices)
        #
        # data = data[~data[dataset_manager.case_id_col].isin(indexes)]
        
        # split data into train and test
        train, test = dataset_manager.split_data(data, train_ratio)
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(datacopy, 0.90))
        del datacopy
        del data

        # create prefix logs
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        print(dt_train_prefixes.shape)
        print(dt_test_prefixes.shape)
        
        # extract arguments
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'n_clusters':None, 
                         'random_state':random_state}
        if bucket_method == "cluster":
            # .rsplit("_", 1)[0]
            bucketer_args['n_clusters'] = best_params[dataset_name][method_name][cls_method]['n_clusters']
        
        cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                            'static_cat_cols':dataset_manager.static_cat_cols,
                            'static_num_cols':dataset_manager.static_num_cols, 
                            'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                            'fillna':fillna}
        
        
        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        #contains list of case_lengths
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            
        pipelines = {}

        results_df = pd.DataFrame()

        # train and fit pipeline for each bucket
        for bucket in set(bucket_assignments_train):
            print("Fitting pipeline for bucket %s..."%bucket)
            
            # set optimal params for this bucket
            if bucket_method == "prefix":
                # print(best_params[dataset_name.rsplit("_",1)[0]][method_name][cls_method][bucket].items())
                cls_args = {k:v for k,v in best_params[dataset_name][method_name][cls_method][str(bucket)].items() if k not in ['n_clusters', 'n_neighbors']}
            else:
                cls_args = {k:v for k,v in best_params[dataset_name][method_name][cls_method].items() if k not in ['n_clusters', 'n_neighbors']}
            cls_args['random_state'] = random_state
            cls_args['min_cases_for_training'] = n_min_cases_in_bucket
            # print(cls_args)

            # select relevant cases
            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event

            # # remove completed cases
            # completed_cases = dt_train_bucket[dt_train_bucket["remtime"] == 0][dataset_manager.case_id_col]
            # dt_train_bucket = dt_train_bucket[~dt_train_bucket[dataset_manager.case_id_col].isin(completed_cases)]

            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            print("Data Sample Before Training...")
            print(dt_train_bucket[["Case ID", "Activity", "Complete Timestamp", "remtime"]])
            print("Training Values...")

            #first transform data using encoding method
            encoder = EncoderFactory.get_encoder(cls_encoding, **cls_encoder_args)
            dt_transformed = encoder.transform(dt_train_bucket)


            
            feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **cls_args))])


            # print("train and target set size", dt_train_bucket.shape, len(train_y))
            pipelines[bucket].fit(dt_train_bucket, train_y)

            # make dataframe to keep track of predictions for each bucket or prefix length and save them in results
            preds_train_bucket = pipelines[bucket].predict_proba(dt_train_bucket).clip(min=0)  # if remaining time is predicted to be negative, make it zero
            print(dt_transformed.shape, len(preds_train_bucket))
            temp = dt_transformed
            #calculcate rae = abs(y_pred - y_true)/(y_true)
            min_true_value = min(train_y.values[np.where(train_y.values!=0)],default=1)
            print("min true val:", min_true_value)
            relative_error = list(map(lambda x: get_rae(x[0],x[1], min_true_value), zip(preds_train_bucket, train_y)))
            temp["true_value"] = np.array(train_y)
            temp["predicted_value"] = np.array(preds_train_bucket)
            temp["relative_error"] = np.array(relative_error)
            print("results being saved for training_bucket..", bucket)
            # /feature_enriched_log_results
            path_res = os.path.abspath('../results/feature_enriched_log_results')
            temp.to_csv(path_res + "/results_train_"+ dataset_ref+"_"+method_name+"_"+cls_method+"_"+str(bucket) +".csv")
            del temp






        prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()
        
        # test separately for each prefix length
        for nr_events in range(min_prefix_length, max_prefix_length+1):
            print("Predicting for %s events..."%nr_events)

            # select only cases that are at least of length nr_events
            relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

            #add code that takes care of certain indices


            if len(relevant_cases_nr_events) == 0:
                break

            dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
            print("Test_nr_events shape", dt_test_nr_events.shape)
            del relevant_cases_nr_events

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

                # # remove completed cases
                # completed_cases = dt_test_bucket[dt_test_bucket["remtime"] == 0][dataset_manager.case_id_col]
                # dt_test_bucket = dt_test_bucket[~dt_test_bucket[dataset_manager.case_id_col].isin(completed_cases)]

                print("test bucket shape", dt_test_bucket.shape)

                if len(relevant_cases_bucket) == 0:
                    continue

                elif bucket not in pipelines: # TODO fix this
                    # use the general class ratio (in training set) as prediction 
                    preds_bucket = array([np.mean(train["remtime"])] * len(relevant_cases_bucket))

                else:
                    # make actual predictions
                    # preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)
                    # encoder = EncoderFactory.get_encoder(cls_encoding, **cls_encoder_args)
                    # dt_test_transformed = encoder.transform(dt_test_bucket).fit()
                    # preds_bucket = pipelines[bucket].predict_proba(dt_test_transformed)

                    preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)


                preds_bucket = preds_bucket.clip(min=0)  # if remaining time is predicted to be negative, make it zero
                preds.extend(preds_bucket)

                # extract actual label values
                test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                test_y.extend(test_y_bucket)

                #write prediction results for test buckets
                encoder = EncoderFactory.get_encoder(cls_encoding, **cls_encoder_args)
                dt_test_transformed = encoder.transform(dt_test_bucket)
                print(dt_test_transformed.shape, len(preds_bucket))
                temp = dt_test_transformed
                # calculcate rae = abs(y_pred - y_true)/(y_true)
                min_true_value = min(test_y_bucket.values[np.where(test_y_bucket.values!=0)], default=1)
                relative_error = list(map(lambda x: get_rae(x[0],x[1], min_true_value), zip(preds_bucket, test_y_bucket)))
                temp["true_value"] = np.array(test_y_bucket)
                temp["predicted_value"] = np.array(preds_bucket)
                temp["relative_error"] = np.array(relative_error)
                print("results being saved for test bucket..", bucket)
                # add / feature_enriched_log_results to path_res when dataset is enriched with features, else remove
                path_res = os.path.abspath('../results/feature_enriched_log_results')
                temp.to_csv(path_res + "/results_test_" +dataset_ref+"_"+method_name+"_"+cls_method+"_"+ str(bucket) + ".csv")
                del temp



            if len(test_y) < 2:
                mae = None
            else:
                # print(preds[:10])
                # print("Ground Truth...")
                # print(test_y[:10])
                mae = mean_absolute_error(test_y, preds)
            #prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "mae", mae, len(test_y)))
            #fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "precision", prec))
            #fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "recall", rec))
            #fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, "fscore", fscore))
            

        print("\n")
