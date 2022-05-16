 import dataset_confs

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences


class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]

        self.scaler = None
        self.encoded_cols = None

    def read_dataset(self):
        # read dataset
        dtypes = {col: "object" for col in self.dynamic_cat_cols + self.static_cat_cols + [self.case_id_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        dtypes[self.label_col] = "float"  # remaining time should be float

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data


    def split_data(self, data, train_ratio):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)


    def generate_prefix_data(self, data, min_length, max_length, comparator):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[comparator(data['case_length'], min_length)].groupby(self.case_id_col).head(min_length)
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[comparator(data['case_length'], nr_events)].groupby(self.case_id_col).head(nr_events)
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes


    def get_case_length_quantile(self, data, quantile=0.90):
        return int(np.floor(data.groupby(self.case_id_col).size().quantile(quantile)))


    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data, mode):
        if self.label_col == "remtime":
            # remtime is a dynamic label (changes with each executed event), take the latest (smallest) value
            return data.groupby(self.case_id_col).min()[self.label_col]
        else:
            # static labels - take any value throughout the case (e.g. the last one)
            return data.groupby(self.case_id_col).last()[self.label_col]
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs / class_freqs.sum()
    
    ### functions for LSTMs ###
    # based on https://github.com/irhete/lstm-predictive-monitoring

    def encode_data(self, data):
        data = data.sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        num_cols = self.dynamic_num_cols + self.static_num_cols
        cat_cols = self.dynamic_cat_cols + self.static_cat_cols
        cat_cols = [col for col in cat_cols if col != self.activity_col]
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
        # dt_all = pd.concat([dt_scaled, dt_cat_act, dt_cat, data[["timesincemidnight", "weekday", "timesincecasestart", "timesincelastevent", "event_nr"]]], axis=1)
        dt_all[self.case_id_col] = data[self.case_id_col]
        dt_all[self.label_col] = data[self.label_col]
        dt_all[self.timestamp_col] = data[self.timestamp_col]

        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = dt_all.columns
        else:
            for col in self.encoded_cols:
                if col not in dt_all.columns:
                    dt_all[col] = 0

        return dt_all[self.encoded_cols]

    def generate_LSTM_data(self, data, max_len):
        data = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col).head(max_len)
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = data.shape[1] - 3

        n_cases = data.shape[0]

        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_o = np.zeros((n_cases, 1), dtype=np.float64)

        idx = 0
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col]
            group = group.to_numpy()
            for i in range(1, len(group) + 1):
                X[idx] = pad_sequences(group[np.newaxis, :i, :-3], maxlen=max_len, dtype=np.float64)
                y_o[idx] = label.iloc[i - 1]
                idx += 1
        return (X, y_o)

    def generate_LSTM_data_prefix_length(self, data, max_len, nr_events):
        grouped = data.groupby(self.case_id_col)

        activity_cols = [col for col in data.columns if col.startswith("act")]
        n_activities = len(activity_cols)
        data_dim = data.shape[1] - 3

        # n_cases = np.sum(grouped.size() > nr_events)
        n_cases = np.sum(grouped.size() >= nr_events)

        # encode only prefixes of this length
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float64)
        y_o = np.zeros((n_cases, 1), dtype=np.float64)
        case_ids = []

        idx = 0
        for case_id, group in grouped:
            if len(group) < nr_events:
                # if len(group) <= nr_events: # in train, use <
                continue
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col].iloc[nr_events - 1]
            group = group.to_numpy()
            X[idx] = pad_sequences(group[np.newaxis, :nr_events, :-3], maxlen=max_len, dtype=np.float64)

            y_o[idx] = label
            case_ids.append(case_id)
            idx += 1

        return (X, y_o, case_ids)

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
