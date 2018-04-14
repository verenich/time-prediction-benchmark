import sys

sys.path.append("..")
import dataset_confs

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences


class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.resource_col = dataset_confs.resource_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        
        self.sorting_cols = [self.timestamp_col, self.activity_col]
        
        if self.dataset_name in dataset_confs.label_col:
            self.label_col = dataset_confs.label_col[self.dataset_name]
        if self.dataset_name in dataset_confs.pos_label:
            self.pos_label = dataset_confs.pos_label[self.dataset_name]
        
        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.scaler = None
        self.encoded_cols = None
        
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.timestamp_col]}
        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes, parse_dates = [self.timestamp_col])
        if "bpic2015" in self.dataset_name:
            data[self.resource_col] = 'res_' + data[self.resource_col].astype(str)

        return data
    
    def extract_timestamp_features(self, data):
        data["timesincemidnight"] = data[self.timestamp_col].dt.hour * 3600 + data[self.timestamp_col].dt.minute * 60 + data[self.timestamp_col].dt.second
        data["weekday"] = data[self.timestamp_col].dt.weekday
        return data
        
    def extract_duration_features(self, data):
        return data.groupby(self.case_id_col).apply(self._extract_durations_for_group)
        
    def _extract_durations_for_group(self, group):
    
        group = group.sort_values(self.sorting_cols, ascending=False, kind="mergesort")

        tmp = group[self.timestamp_col] - group[self.timestamp_col].shift(-1)
        tmp = tmp.fillna(0)
        group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's'))) # m is for minutes

        tmp = group[self.timestamp_col] - group[self.timestamp_col].iloc[-1]
        tmp = tmp.fillna(0)
        group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 's'))) # m is for minutes

        group = group.sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        group["event_nr"] = range(1, len(group) + 1)

        return group


    def split_data(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, test)
    
    def split_data_strict(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)
    
    def split_val(self, data, val_ratio, split="random", seed=11):
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[:int(val_ratio*len(start_timestamps))]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)
    
    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.floor(data.groupby(self.case_id_col).size().quantile(quantile)))

    def get_train_val_data(self, data, sample_size=None, val_sample_size=None, train_ratio=0.8):
        # adjust sample sizes if the total number of cases is smaller than required
        grouped = data.groupby(self.case_id_col)
        if sample_size is None or val_sample_size is None or val_sample_size + sample_size > len(grouped):
            sample_size = int(len(grouped) * train_ratio)
            val_sample_size = len(grouped) - sample_size
            print("Adjusted sample sizes, new sample_size: ", sample_size, ", new val_sample_size: ", val_sample_size)
        
        case_ids = [name for name, _ in grouped]
        train = data[data[self.case_id_col].isin(case_ids[:sample_size])]
        val = data[data[self.case_id_col].isin(case_ids[sample_size:sample_size+val_sample_size])]
        return (train, val)
    
    def get_train_sample(self, data, sample_size=None):
        # adjust sample sizes if the total number of cases is smaller than required
        grouped = data.groupby(self.case_id_col)
        if sample_size is None or sample_size > len(grouped):
            print("Adjusted sample size, new sample_size: ", len(grouped))
            return(data)
        
        case_ids = [name for name, _ in grouped]
        train = data[data[self.case_id_col].isin(case_ids[:sample_size])]
        return train
    
    def encode_data(self, data):
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        # extract time features
        data = self.extract_timestamp_features(data)
        data = self.extract_duration_features(data)

        # one-hot encode activity col
        data = pd.get_dummies(data, columns=[self.activity_col], prefix="act")
        
        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = data.columns
        else:
            for col in self.encoded_cols:
                if col not in data.columns:
                    data[col] = 0
        
        return data[self.encoded_cols]
    
    def encode_data_with_label(self, data):
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        # extract time features
        data = self.extract_timestamp_features(data)
        data = self.extract_duration_features(data)

        # one-hot encode activity col
        data = pd.get_dummies(data, columns=[self.activity_col], prefix="act")
        #data[self.label_col] = data[self.label_col].apply(lambda x: 1 if x == self.pos_label else 0)

        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = data.columns
        else:
            for col in self.encoded_cols:
                if col not in data.columns:
                    data[col] = 0

        return data[self.encoded_cols]

    def encode_data_with_label_all_data(self, data, extract_timestamp_features=False, extract_duration_features=False):
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        # extract time features
        if extract_timestamp_features:
            data = self.extract_timestamp_features(data)
        if extract_duration_features:
            data = self.extract_duration_features(data)

        num_cols = self.dynamic_num_cols+self.static_num_cols
        #num_cols = [col for col in num_cols if col not in ["duration", "weekday", "hour"]]
        cat_cols = self.dynamic_cat_cols+self.static_cat_cols
        cat_cols = [col for col in cat_cols if col!= self.activity_col]
        # scale numeric cols
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            dt_scaled = pd.DataFrame(self.scaler.fit_transform(data[num_cols]), index=data.index, columns=num_cols)
        else:
            dt_scaled = pd.DataFrame(self.scaler.transform(data[num_cols]), index=data.index, columns=num_cols)

        # one-hot encode categorical cols
        dt_cat_act = pd.get_dummies(data[self.activity_col], columns=[self.activity_col], prefix="act")
        dt_cat = pd.get_dummies(data[cat_cols])

        # merge
        dt_all = pd.concat([dt_scaled, dt_cat_act, dt_cat], axis=1)
        #dt_all = pd.concat([dt_scaled, dt_cat_act, dt_cat, data[["timesincemidnight", "weekday", "timesincecasestart", "timesincelastevent", "event_nr"]]], axis=1)
        dt_all[self.case_id_col] = data[self.case_id_col]
        dt_all[self.label_col] = data[self.label_col] #.apply(lambda x: 1 if x == self.pos_label else 0)
        dt_all[self.timestamp_col] = data[self.timestamp_col]

        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = dt_all.columns
        else:
            for col in self.encoded_cols:
                if col not in dt_all.columns:
                    dt_all[col] = 0

        return dt_all[self.encoded_cols]


    def generate_3d_data(self, data, max_len):
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = n_activities + 5
        time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
        relevant_cols = activity_cols + ["event_nr"] + time_cols

        n_cases = data.shape[0]

        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)

        idx = 0
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")[relevant_cols].as_matrix()
            for i in range(1, len(group) + 1):
                row = group[np.newaxis,:i,:]
                X[idx] = pad_sequences(row, maxlen=max_len, dtype=np.float64)
                # add last event indicator
                if i == len(group):
                    y_a[idx,-1] = 1
                    y_t[idx] = 0
                else:
                    y_a[idx,:-1] = group[np.newaxis,i,:n_activities]
                    y_t[idx] = group[np.newaxis,i,n_activities + 1]
                idx += 1
        return (X, y_a, y_t)

    def generate_3d_data_for_timedistributed(self, data, max_len, n_classes=2):
        n_samples = data.shape[0]
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        X = np.empty((n_samples, max_len, data.shape[1] - 3), dtype=np.float64)
        y = np.zeros((n_samples, max_len, n_classes), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            label = group[self.label_col].iloc[0]
            group = group.as_matrix()
            X[idx,:,:] = pad_sequences(group[np.newaxis,:max_len,:-3], maxlen=max_len)
            y[idx,:,label] = 1 # y[:,:,0] stands for negative class, y[:,:,1] for positive
            idx += 1
        return (X, y)

    def generate_3d_data_for_timedistributed_prefixes(self, data, max_len, n_classes=2):
        data = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col).head(max_len)
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        data_dim = data.shape[1] - 3
        n_cases = data.shape[0]

        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y = np.zeros((n_cases, max_len, n_classes), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col].iloc[0]
            group = group.as_matrix()
            for i in range(1, len(group) + 1):
                X[idx] = pad_sequences(group[np.newaxis,:i,:-3], maxlen=max_len, dtype=np.float64)
                # add last event indicator
                y[idx,:,label] = 1
                idx += 1
        return (X, y)

    def generate_3d_data_with_label(self, data, max_len):
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = n_activities + 5
        time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
        relevant_cols = activity_cols + ["event_nr"] + time_cols

        n_cases = data.shape[0]

        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)
        y_o = np.zeros((n_cases, 2), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = [group[self.label_col].iloc[0], 1-group[self.label_col].iloc[0]]
            group = group[relevant_cols].as_matrix()
            for i in range(1, len(group) + 1):
                row = group[np.newaxis,:i,:]
                X[idx] = pad_sequences(row, maxlen=max_len, dtype=np.float64)
                # add last event indicator
                if i == len(group):
                    y_a[idx,-1] = 1
                    y_t[idx] = 0
                else:
                    y_a[idx,:-1] = group[np.newaxis,i,:n_activities]
                    y_t[idx] = group[np.newaxis,i,n_activities + 1]
                y_o[idx] = label
                idx += 1
        return (X, y_a, y_t, y_o)

    def generate_3d_data_with_label_all_data(self, data, max_len):
        data = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col).head(max_len)
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = data.shape[1] - 3

        n_cases = data.shape[0]

        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)
        y_o = np.zeros((n_cases, 1), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col]
            group_activity = group[activity_cols].as_matrix()
            #group_time = group["timesincelastevent"].as_matrix()
            group = group.as_matrix()
            for i in range(1, len(group) + 1):
                X[idx] = pad_sequences(group[np.newaxis,:i,:-3], maxlen=max_len, dtype=np.float64)
                # add last event indicator
                """
                if i == len(group):
                    y_a[idx,-1] = 1
                    y_t[idx] = 0
                else:
                    y_a[idx,:-1] = group_activity[np.newaxis,i,:]
                    y_t[idx] = group_time[np.newaxis,i]
                """
                y_o[idx] = label.iloc[i-1]
                idx += 1
        print("here!")
        return (X, y_a, y_t, y_o)

    def data_generator(self, data, max_len):
        data = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col).head(max_len)
        print("Generator: %s" % len(data.groupby(self.case_id_col)))
        data_dim = data.shape[1] - 3
        while 1:
            grouped = data.groupby(self.case_id_col)
            for _, group in grouped:
                n_cases = len(group)
                X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
                y_o = np.zeros((n_cases, 2), dtype=np.float64)
                idx = 0

                group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
                label = [group[self.label_col].iloc[0], 1-group[self.label_col].iloc[0]]
                group = group.as_matrix()
                for i in range(1, len(group) + 1):
                    X[idx] = pad_sequences(group[np.newaxis,:i,:-3], maxlen=max_len, dtype=np.float64)
                    y_o[idx] = label
                    idx += 1
                yield {'main_input': X}, {'outcome_output': y_o}


    def generate_3d_data_for_prefix_length(self, data, max_len, nr_events):
        grouped = data.groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = n_activities + 5
        time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
        relevant_cols = activity_cols + ["event_nr"] + time_cols

        n_cases = np.sum(grouped.size() > nr_events)

        # encode only prefixes of this length
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)

        idx = 0
        for _, group in grouped:
            if len(group) <= nr_events: # in train, use <
                continue

            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")[relevant_cols].as_matrix()
            row = group[np.newaxis,:nr_events,:]
            X[idx] = pad_sequences(row, maxlen=max_len, dtype=np.float64)
            # add last event indicator in train set, can't happen in testing phase
            if len(group) == nr_events:
                y_a[idx,-1] = 1
                y_t[idx] = 0
            else:
                y_a[idx,:-1] = group[np.newaxis,nr_events,:n_activities]
                y_t[idx] = group[np.newaxis,nr_events,n_activities + 1]
            idx += 1

        return (X, y_a, y_t)

    def generate_3d_data_for_prefix_length_with_label(self, data, max_len, nr_events):
        grouped = data.groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = n_activities + 5
        time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
        relevant_cols = activity_cols + ["event_nr"] + time_cols

        n_cases = np.sum(grouped.size() > nr_events)

        # encode only prefixes of this length
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)
        y_o = np.zeros((n_cases, 2), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            if len(group) <= nr_events: # in train, use <
                continue
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = [group[self.label_col].iloc[0], 1-group[self.label_col].iloc[0]]
            group = group[relevant_cols].as_matrix()
            row = group[np.newaxis,:nr_events,:]
            X[idx] = pad_sequences(row, maxlen=max_len, dtype=np.float64)
            # add last event indicator in train set, can't happen in testing phase
            if len(group) == nr_events:
                y_a[idx,-1] = 1
                y_t[idx] = 0
            else:
                y_a[idx,:-1] = group[np.newaxis,nr_events,:n_activities]
                y_t[idx] = group[np.newaxis,nr_events,n_activities + 1]
            y_o[idx] = label
            idx += 1

        return (X, y_a, y_t, y_o)

    def generate_3d_data_for_prefix_length_with_label_all_data(self, data, max_len, nr_events):
        grouped = data.groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = data.shape[1] - 3

        #n_cases = np.sum(grouped.size() > nr_events)
        n_cases = np.sum(grouped.size() >= nr_events)

        # encode only prefixes of this length
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)
        y_o = np.zeros((n_cases, 1), dtype=np.float64)
        case_ids = []

        idx = 0
        for case_id, group in grouped:
            if len(group) < nr_events:
            #if len(group) <= nr_events: # in train, use <
                continue
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col].iloc[nr_events-1]
            group_activity = group[activity_cols].as_matrix()
            #group_time = group["timesincelastevent"].as_matrix()
            group = group.as_matrix()
            X[idx] = pad_sequences(group[np.newaxis,:nr_events,:-3], maxlen=max_len, dtype=np.float64)
            # add last event indicator in train set, can't happen in testing phase
            """
            if len(group) == nr_events:
                y_a[idx,-1] = 1
                y_t[idx] = 0
            else:
                y_a[idx,:-1] = group_activity[np.newaxis,nr_events,:]
                y_t[idx] = group_time[np.newaxis,nr_events]
            """
            y_o[idx] = label
            case_ids.append(case_id)
            idx += 1

        return (X, y_a, y_t, y_o, case_ids)
    
    def generate_3d_data_for_prefix_length_no_padding(self, data, nr_events, mode="train"):
        grouped = data.groupby(self.case_id_col)
        
        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = n_activities + 5
        time_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "weekday"]
        relevant_cols = activity_cols + ["event_nr"] + time_cols
        
        if mode == "train":
            n_cases = np.sum(grouped.size() >= nr_events)
        else:
            n_cases = np.sum(grouped.size() > nr_events)
        
        # encode only prefixes of this length
        X = np.zeros((n_cases, nr_events, data_dim), dtype=np.float64)
        y_a = np.zeros((n_cases, n_activities + 1), dtype=np.int32)
        y_t = np.zeros(n_cases, dtype=np.float64)
        
        idx = 0
        for _, group in grouped:
            if len(group) < nr_events: # in train, use <
                continue
            elif len(group) == nr_events and mode == "test":
                continue

            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")[relevant_cols].as_matrix()
            X[idx] = group[np.newaxis,:nr_events,:]
            # add last event indicator in train set, can't happen in testing phase
            if len(group) == nr_events:
                y_a[idx,-1] = 1
                y_t[idx] = 0
            else:
                y_a[idx,:-1] = group[np.newaxis,nr_events,:n_activities]
                y_t[idx] = group[np.newaxis,nr_events,n_activities + 1]
            idx += 1
        
        return (X.reshape((X.shape[0], X.shape[1] * X.shape[2])), y_a, y_t)

    def get_max_case_length(self, data):
        return data.groupby(self.case_id_col).size().max()

    def calculate_divisors(self, data):
        self.divisors = {}
        self.divisors["timesincelastevent"] = np.mean(data["timesincelastevent"])
        self.divisors["timesincecasestart"] = np.mean(data["timesincecasestart"])
        self.divisors["timesincemidnight"] = 86400.0
        self.divisors["weekday"] = 7.0

    def normalize_data(self, data):
        for col, divisor in self.divisors.items():
            data[col] = data[col] / divisor
        return data


    
