import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import random
import pdb
import warnings
import pickle
import glob


warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 cycle=None, use_cycle=False):
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.use_cycle = use_cycle 

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else: 
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.use_cycle and self.cycle is not None:
            self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_cycle and self.cycle is not None:
            cycle_index = torch.tensor(self.cycle_index[s_end])
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cycle=None,  use_cycle=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.use_cycle = use_cycle  

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        if self.use_cycle and self.cycle is not None:
            self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_cycle and self.cycle is not None:
            cycle_index = torch.tensor(self.cycle_index[s_end])
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None, use_cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.use_cycle = use_cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.use_cycle and self.cycle is not None:
            self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
            
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_cycle and self.cycle is not None:
            cycle_index = torch.tensor(self.cycle_index[s_end])
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark



    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False,
                 timeenc=0, freq='15min', cols=None,
                 cycle=None, use_cycle=False):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag == 'pred'

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.cycle = cycle
        self.use_cycle = use_cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Compute cycle_index if needed
        if self.use_cycle and self.cycle is not None:
            self.cycle_index = (np.arange(len(self.data_x) + self.pred_len) % self.cycle)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_cycle and self.cycle is not None:
            cycle_idx = torch.tensor(self.cycle_index[s_end])
            return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_idx
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ODE_updated(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ode/raw_data/electrophysiology/inada_N_2009/T=10_ds=0.5_dc=0.05',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False,
                 cycle=None, use_cycle=False):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.cycle = cycle
        self.use_cycle = use_cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def load_dataframes(self, folder_path):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder {folder_path} does not exist")
        file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
        if not file_paths:
            raise ValueError("No CSV files found in folder")
        return [pd.read_csv(f) for f in file_paths]

    def __read_data__(self):
        self.scaler = StandardScaler()
        csv_files = self.load_dataframes(os.path.join(self.root_path, self.data_path))

        n_files = len(csv_files)
        train_files = csv_files[:int(0.7 * n_files)]
        val_files = csv_files[int(0.7 * n_files):int(0.8 * n_files)]
        test_files = csv_files[int(0.8 * n_files):]

        def create_windows(file_list):
            windows, indices = [], []
            time_index = 0
            total_length = self.seq_len + self.pred_len
            for file in file_list:
                file_data = file.values
                for i in range(len(file_data) - total_length + 1):
                    window = file_data[i:i+total_length]
                    x, y = window[:self.seq_len, 1:], window[self.seq_len:, 1:]
                    x_mark, y_mark = window[:self.seq_len, 0], window[self.seq_len:, 0]
                    windows.append((x, y, x_mark, y_mark))
                    indices.append(time_index + self.seq_len)
                    time_index += 1
                time_index += total_length  # avoid overlapping between files
            return windows, indices

        self.data = {}
        self.indices = {}
        self.data["train"], self.indices["train"] = create_windows(train_files)
        self.data["val"], self.indices["val"] = create_windows(val_files)
        self.data["test"], self.indices["test"] = create_windows(test_files)

        train_sequences = np.concatenate([x[0] for x in self.data["train"]], axis=0)

        # Store data with or without cycle_index
        if self.scale:
            self.scaler.fit(train_sequences)

        if self.use_cycle and self.cycle is not None:
            self.data_x = [
                (self.scaler.transform(x[0]) if self.scale else x[0],
                 self.scaler.transform(x[1]) if self.scale else x[1],
                 x[2], x[3], idx % self.cycle)
                for x, idx in zip(self.data[self.flag], self.indices[self.flag])
            ]
        else:
            self.data_x = [
                (self.scaler.transform(x[0]) if self.scale else x[0],
                 self.scaler.transform(x[1]) if self.scale else x[1],
                 x[2], x[3])
                for x in self.data[self.flag]
            ]

    def __getitem__(self, index):
        if self.use_cycle and self.cycle is not None:
            seq_x, seq_y, x_mark, y_mark, cycle_index = self.data_x[index]
            return seq_x, seq_y, x_mark, y_mark, torch.tensor(cycle_index)
        else:
            seq_x, seq_y, x_mark, y_mark = self.data_x[index]
            return seq_x, seq_y, x_mark, y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ODE(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ode/raw_data/electrophysiology/inada_N_2009/T=10_ds=0.5_dc=0.05',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # load all the files

        train_windows = []
        val_windows = []
        test_windows = [] 

        for f in range(100):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path+"/ts-data-"+str(f)+".csv")) 
            df_raw = df_raw.drop(columns=["t"]) # [1500 rows x 29 columns] + 1 time
            data = df_raw.values # (1500, 29)
            t,c  = data.shape
            # split the dataset in train test and val similar to custom data split but remove the seq_len so that the windows are 
            #  distributed in 70 10 20 ratio. 
            train_end = int(0.7*(t-self.seq_len)) #.7*204
            val_end = train_end + int(0.15*(t-self.seq_len)) #.1*204
            test_end = val_end + int(0.15*(t-self.seq_len)) #.2*204

            # print(train_end, val_end, test_end) # 142 162 202

            train_end = self.seq_len+ train_end
            val_end = self.seq_len+ val_end
            test_end = self.seq_len+ test_end

            # print(train_end, val_end, test_end) # 238 258 298

            assert train_end+self.pred_len <= val_end, "prediction len too large"

            # get train windows
            for index in range(self.seq_len, train_end - self.pred_len): # 238-12 -96 = 130
                s_end  = index  
                s_begin  = index - self.seq_len
                r_begin = index
                r_end = index + self.pred_len
                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                seq_x_mark = data[s_begin:s_end]
                seq_y_mark = data[r_begin:r_end]
                train_windows.append((seq_x, seq_y, seq_x_mark, seq_y_mark))

            # get val windows 
            for index in range(train_end, val_end - self.pred_len): # 258-12 -238  = 246 - 238 = 8
                s_end  = index
                s_begin  = index - self.seq_len
                r_begin = index
                r_end = index + self.pred_len
                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                seq_x_mark = data[s_begin:s_end]
                seq_y_mark = data[r_begin:r_end]
                val_windows.append((seq_x, seq_y, seq_x_mark, seq_y_mark))

            # get test windows 
            for index in range(val_end, test_end - self.pred_len): # 298-12 -258 -> 300-24 = 28
                s_end  = index
                s_begin  = index - self.seq_len
                r_begin = index
                r_end = index + self.pred_len
                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                seq_x_mark = data[s_begin:s_end]
                seq_y_mark = data[r_begin:r_end]
                test_windows.append((seq_x, seq_y, seq_x_mark, seq_y_mark))
                
        self.data = {}

        self.data["train"] = train_windows
        self.data["val"] = val_windows
        self.data["test"] = test_windows

        train_sequences = np.concatenate([x[0] for x in self.data["train"]], axis=0)

        ## scale the dataset  
        if self.scale:
            self.scaler.fit(train_sequences)
            # print(self.scaler.mean_)
            self.data_x = [(self.scaler.transform(x[0]), self.scaler.transform(x[1]), self.scaler.transform(x[2]), self.scaler.transform(x[3])) for x in self.data[self.flag]] # see if we have a better method
        else:
            self.data_x = self.data[self.flag] 



        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

    def __getitem__(self, index):
        
        return self.data_x[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PDE(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='pde/diffs_sorp/100_csvs',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # load all the files
        windows = [] 
        for f in range(100):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path+"/"+str(f).zfill(4)+".csv"), index_col=0)
            df_raw = df_raw.drop(columns=["t"])
            data = df_raw.values
            for index in range(self.seq_len, len(df_raw)- self.pred_len):
                s_end  = index
                s_begin  = index - self.seq_len
                r_begin = index
                r_end = index + self.pred_len
                seq_x = data[s_begin:s_end]
                seq_y = data[r_begin:r_end]
                seq_x_mark = data[s_begin:s_end]
                seq_y_mark = data[r_begin:r_end]
                windows.append((seq_x, seq_y, seq_x_mark, seq_y_mark))


        # split this into train val test split

        def select_random_percent(values, threshold):
            sample_size = int(len(values) * threshold)
            selected_values = random.sample(values, sample_size)
            remaining_values = [value for value in values if value not in selected_values]
            return selected_values, remaining_values

        possible_windows = list(range(len(windows)))
        test_samples, train_samples = select_random_percent(possible_windows, 0.2) # 0.2 test
        val_samples, train_samples = select_random_percent(train_samples, 0.2) # 0.2 val

        print("test : ", len(test_samples), " val :", len(val_samples), " train : ", len(train_samples))

        self.data = {}

        self.data["train"] = [windows[i] for i in train_samples]
        self.data["val"] = [windows[i] for i in val_samples]
        self.data["test"] = [windows[i] for i in test_samples]

        # pdb.set_trace()
        train_sequences = np.concatenate([x[0] for x in self.data["train"]], axis=0)

        ## scale the dataset  
        if self.scale:
            self.scaler.fit(train_sequences)
            # print(self.scaler.mean_)
            self.data_x = [(self.scaler.transform(x[0]), self.scaler.transform(x[1]), self.scaler.transform(x[2]), self.scaler.transform(x[3])) for x in self.data[self.flag]] # see if we have a better method
        else:
            self.data_x = self.data[self.flag] 



        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''


    def __getitem__(self, index):
        
        return self.data_x[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)