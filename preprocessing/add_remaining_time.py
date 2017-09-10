import pandas as pd
import numpy as np
import os

input_data_folder = "../experiments/labeled_logs_csv"
output_data_folder = "../experiments/labeled_logs_csv_processed"

filenames_bpic2011 = ["BPIC11.csv"]
filenames_bpic2015 = ["BPIC15_%s.csv"%municipality for municipality in range(1,6)]
filenames_bpic2017 = ["BPIC17.csv"]
filenames_others = ["traffic_fines.csv", "Sepsis.csv"]
#filenames = filenames_bpic2011 + filenames_bpic2015 + filenames_bpic2017 + filenames_others
filenames = ["traffic_fines.csv"]
timestamp_col = "Complete Timestamp"
label_col = "label"
case_id_col = "Case ID"

def add_remtime_column(group):
    group = group.sort_values(timestamp_col, ascending=False)
    end_date = group[timestamp_col].iloc[0]

    tmp = end_date - group[timestamp_col]
    tmp = tmp.fillna(0)
    group["remtime"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's'))) # s is for seconds

    return group

for filename in filenames:
    print(filename)
    data = pd.read_csv(os.path.join(input_data_folder, filename), sep=";")
    data = data.drop([label_col], axis=1)
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.groupby(case_id_col).apply(add_remtime_column)
    data.to_csv(os.path.join(output_data_folder, filename), sep=";", index=False)





