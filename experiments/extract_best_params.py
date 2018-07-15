import pandas as pd
import os
import pickle
import json


datasets = ["production", "credit", "bpic2012o", "bpic2012w", "credit",
            "helpdesk", "hospital", "invoice", "sepsis", "bpic2011",
            "bpic2015", "bpic2017", "traffic_fines"]

bucket_methods = ["single", "state", "cluster"]
cls_encodings = ["laststate", "agg", "index"]
cls_methods = ["xgb"]


for dataset_name in datasets:
    for bucket_method in bucket_methods:
        for cls_encoding in cls_encodings:
            for cls_method in cls_methods:
                optimal_params_filename = os.path.join("optimal_params", "%s_%s_%s_%s" % (cls_method, dataset_name, bucket_method, cls_encoding))
                if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                    file = '../results/CV/param_optim_%s_%s_%s_%s.csv' % (cls_method, dataset_name, bucket_method, cls_encoding)
                    if not os.path.isfile(file) or os.path.getsize(file) <= 52:
                        print(file)
                        continue
                    data = pd.read_csv(file, sep=";")
                    best_params = {val[0]: val[1] for _, val in data[data.score==data[data.param!="processing_time"].score.min()][["param", "value"]].iterrows()}
                    
                    # write to file
                    with open("%s.pkl" % optimal_params_filename, "wb") as fout:
                        pickle.dump(best_params, fout)

                    with open("%s.json" % optimal_params_filename, "w") as fout:
                        json.dump(best_params, fout, indent=3)


bucket_methods = ["prefix"]
cls_encodings = ["laststate", "agg", "index"]


for dataset_name in datasets:
    for bucket_method in bucket_methods:
        for cls_encoding in cls_encodings:
            for cls_method in cls_methods:
                optimal_params_filename = os.path.join("optimal_params", "%s_%s_%s_%s" % (cls_method, dataset_name, bucket_method, cls_encoding))
                file = '../results/CV/param_optim_%s_%s_%s_%s.csv' % (cls_method, dataset_name, bucket_method, cls_encoding)
                if os.path.isfile(file) and os.path.getsize(file) > 0:
                    data = pd.read_csv(file, sep=";")
                    data = data[data.param!="processing_time"]
                    best_params = {}
                    if "nr_events" in list(data.columns):
                        for nr_events, group in data.groupby("nr_events"):
                            vals = {val[0]: val[1] for _, val in list(group[group.score==group.score.min()].groupby("iter"))[0][1][["param", "value"]].iterrows()}
                            best_params[nr_events] = vals
                        
                        # write to file
                        with open("%s.pkl" % optimal_params_filename, "wb") as fout:
                            pickle.dump(best_params, fout)

                        with open("%s.json" % optimal_params_filename, "w") as fout:
                            json.dump(best_params, fout, indent=3)
