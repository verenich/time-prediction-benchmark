import glob
import pandas as pd
import numpy as np
import os


filenames = glob.glob("*.csv")
filenames = [filename for filename in filenames if os.path.getsize(filename) > 10000]
#filenames = ["CreditRequirement.csv"]

timestamp_col = "Complete Timestamp" # column that indicates completion timestamp
case_id_col = "Case ID"
activity_col = "Activity"



def add_all_columns(group):
    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort")
    group["event_nr"] = range(1,group.shape[0]+1)
    group["unique_events"] = group[activity_col].nunique()
    group["total_events"] = len(group[activity_col])
    end_date = group[timestamp_col].iloc[-1]
    tmp = end_date - group[timestamp_col]
    tmp = tmp.fillna(0)
    start_date = group[timestamp_col].iloc[0]
    elapsed = group[timestamp_col] - start_date
    elapsed = elapsed.fillna(0)
    group["elapsed"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 'D')))
    group["remtime"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'D')))  # D is for days
    #group["case_length"] = group.shape[0]
    return group



with open("log_summary.tsv", 'w') as fout:
    fout.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
    "log", "total_cases", "unique_activities", "total_events","avg_unique_events_per_trace", "mean_case_length",
    "std_case_length", "mean_case_duration","std_case_duration","mean_remtime","std_remtime"))
    for filename in filenames:
        print(filename)
        # dtypes = {col:"str" for col in ["proctime", "elapsed", "label", "last"]} # prevent type coercion
        data = pd.read_csv(filename, sep=";")
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        data = data.groupby(case_id_col).apply(add_all_columns)
        df0 = data.loc[data["event_nr"] == 1].copy()
        df0["UER"] = df0["unique_events"] / df0["total_events"]
        #print("Avg percentage of unique timestamps per trace: %.3f" %np.mean(df0["UTR"]))
        #print("%s out of %s unique timestamps" %(len(data[timestamp_col].unique()),data[timestamp_col].count()))
        global_unique_timestamps = len(data[timestamp_col].unique()) / data[timestamp_col].count()
        #print("%s cases that reach length %d" %(df.shape[0],cutoff))
        #print("In %s of them elapsed time is still 0" %len(df.loc[df["elapsed"]==0]))
        #print("%s cases that reach length %d" %(df.shape[0],cutoff))
        fout.write("%s, %s, %s, %s, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n"%(filename,
                                                                                 data[case_id_col].nunique(),
                                                                                 data[activity_col].nunique(),
                                                                                 data.shape[0],
                                                                                 np.mean(df0["UER"]),
                                                                                 np.mean(df0["total_events"]),
                                                                                 np.std(df0["total_events"]),
                                                                                 np.mean(df0["remtime"]),
                                                                                 np.std(df0["remtime"]),
                                                                                 np.mean(data["remtime"]),
                                                                                 np.std(data["remtime"])))

