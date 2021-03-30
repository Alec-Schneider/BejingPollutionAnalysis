from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
import pandas as pd
import os
import glob
import datetime
import warnings
# pytorch imports
import torch
from torch.utils.data.sampler import SequentialSampler, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

#sklearn
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# get the device available
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Preprocessor:
    def __init__(self, **kwargs):
        self._filtered_vals = None
        self._sigma = None

    def create_masked_array(self, array):
        X = ma.asarray(array)
        # Mask missing data points
        X[np.isnan(X)] = ma.masked
        return X
    
    def time_step(self, index):
        return (index[1] - index[0]).days


    def first_data_index(self, array):
        index_array = np.argwhere(~np.isnan(array))
        index = index_array[0][0]
        return index

    
    def kalman_estimation(self, array, index):
        dt = self.time_step(index)
        X = array.copy()

        n_timesteps = len(index)
        n_dim_state = 3

        # transition_matrix  
        F = [[1,  dt, 0.5*dt*dt], 
            [0,    1,         dt],
            [0,    0,         1]]  

        # observation_matrix   
        H = [1, 0, 0]

        # transition_covariance 
        Q = [[   1,     0,     0], 
            [   0,  1e-4,     0],
            [   0,     0,  1e-6]] 

        # observation_covariance 
        R = [0.04] 

        # initial_state_mean
        X0 = [0,
              0,
              0]

        # initial_state_covariance
        P0 = [[ 10,    0,   0], 
            [  0,    1,   0],
            [  0,    0,   1]]

        self.filtered_state_means = np.zeros((n_timesteps, n_dim_state))
        self.filtered_state_covariances = np.zeros((n_timesteps, n_dim_state,n_dim_state))

        # Kalman-Filter initialization
        kf = KalmanFilter(transition_matrices=F, 
                  observation_matrices=H, 
                  transition_covariance=Q, 
                  observation_covariance=R, 
                  initial_state_mean=X0, 
                  initial_state_covariance=P0)
        
        # iterative estimation for each new measurement
        for t in range(n_timesteps):
            if t == 0:
                self.filtered_state_means[t] = X0
                self.filtered_state_covariances[t] = P0
            else:
                self.filtered_state_means[t], self.filtered_state_covariances[t] = (
                kf.filter_update(
                    self.filtered_state_means[t-1],
                    self.filtered_state_covariances[t-1],
                    observation = X[t])
                )
        filtered_values = self.filtered_state_means[:, 0]
        sigma = np.sqrt(self.filtered_state_covariances[:, 0, 0])
        
        return filtered_values, sigma
    
    def replace_nans(self, orig, filtered_values):
        new = orig.copy()
        np.putmask(new, np.isnan(new), filtered_values)
        return new.data

    def kalman_impute(self, series):
        # Find first non-missing data point and get the index
        first = self.first_data_index(series.values)
        # reset the series to 
        series = series[first:]
        index = series.index
        arr = self.create_masked_array(series)
        self._filtered_vals, self._sigma = self.kalman_estimation(arr, index)
        # replace missing values with Kalman filter prediction
        filled_arr = self.replace_nans(arr, self._filtered_vals)

        filled_series = pd.Series(data=filled_arr,
                                index=index)
        return filled_series


def to_device(data, device=None):
    """Move tensor(s) to chosen device"""
    if not device:
        device = _device

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def split_indicies(n, val_pct):
    n_val = 1 - int(val_pct * n)
    n = np.arange(n)
    return n[:n_val], n[n_val:]


def get_dataloader(dataset, val_pct=None, batch_size=1, 
                    num_workers=0, shuffle=False, device=None):
    """
    Return a DeviceDataLoader object
    """

    if val_pct:
        train_indicies, val_indicies = split_indicies(len(dataset), val_pct)
        print("train indicies:", train_indicies[:10])
        print("val indicies:", val_indicies[:10])
    else:
        train_indicies = np.arange(len(dataset))
        print("train indicies:", train_indicies[:10])

    train_sampler = SubsetSequentialSampler(train_indicies)
    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler, num_workers=num_workers,
            shuffle=shuffle)
    train_loader = DeviceDataLoader(train_loader, device=device)

    if val_pct:
        val_sampler = SubsetSequentialSampler(val_indicies)
        val_loader = DataLoader(dataset, batch_size, sampler=val_sampler, num_workers=num_workers,
            shuffle=shuffle)
        val_loader = DeviceDataLoader(val_loader, device=device)
        
        return train_loader, val_loader

    return train_loader

def get_dataloader2(dataset, batch_size=1, 
                    num_workers=0, shuffle=False, device=None):
    """
    Return a DeviceDataLoader object
    """

    # if val_pct:
    #     train_indicies, val_indicies = split_indicies(len(dataset), val_pct)
    #     print("train indicies:", train_indicies[:10])
    #     print("val indicies:", val_indicies[:10])
    # else:
    #     train_indicies = np.arange(len(dataset))
    #     print("train indicies:", train_indicies[:10])

    #train_sampler = SubsetSequentialSampler(train_indicies)
    train_loader = DataLoader(dataset, batch_size, num_workers=num_workers,
            shuffle=shuffle)
    train_loader = DeviceDataLoader(train_loader, device=device)

    # if val_pct:
    #     #val_sampler = SubsetSequentialSampler(val_indicies)
    #     val_loader = DataLoader(dataset, batch_size, num_workers=num_workers,
    #         shuffle=shuffle)
    #     val_loader = DeviceDataLoader(val_loader, device=device)
        
    #     return train_loader, val_loader

    return train_loader


class SubsetSequentialSampler(Sampler):
    """
    Subeset a data set sequentially
    """
    def __init__(self, indicies):
        self.indicies = indicies

    def __iter__(self):
        return (
            self.indicies[i] for i in torch.arange(0, len(self.indicies))
        )

    def __len__(self):
        return len(self.indicies)


class DeviceDataLoader():
    "Wrap a dataloader to move data to a device"
    def __init__(self, dl, device=None):
        self.dl = dl
        if device:
            device = device
        else:
            device = _device

        self.device = device

    def __iter__(self):
        """
        Yield a batch of data after moving it to device
        """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """
        Number of batches
        """
        return len(self.dl)


class BejingAirDataset(Dataset):
    def __init__(
        self,
        root,
        train = True):

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        
        self.root = root
        self.train = train
        self.features = ['num__PM10', 'num__SO2', 'num__NO2', 'num__CO', 'num__O3',
       'num__TEMP', 'num__PRES', 'num__DEWP', 'num__RAIN', 'num__WSPM',
       'cos_sin__Day sin', 'cos_sin__Day cos', 'cos_sin__Year sin',
       'cos_sin__Year cos', 'cat__x0_E', 'cat__x0_ENE', 'cat__x0_ESE',
       'cat__x0_N', 'cat__x0_NE', 'cat__x0_NNE', 'cat__x0_NNW', 'cat__x0_NW',
       'cat__x0_S', 'cat__x0_SE', 'cat__x0_SSE', 'cat__x0_SSW', 'cat__x0_SW',
       'cat__x0_W', 'cat__x0_WNW', 'cat__x0_WSW']
        self.target = ["num__PM2.5"]
        self.cache = {}

        if self.train:
            self.files = glob.glob(os.path.join(self.train_folder, "*.csv"))
        else:
            self.files = glob.glob(os.path.join(self.test_folder, "*.csv"))
        print("self.files:", self.files)

    @property
    def train_folder(self):
        return os.path.join(self.root, "train")
    
    @property
    def test_folder(self):
        return os.path.join(self.root, "test")


    def __getitem__(self, index, index_col=0):
        #print(index)
        #self.data = self.cache.get(index, None)

        data_path = self.files[index]
        #print('fetching... {0}'.format(data_path))

        #if self.data is None:
        # Read in the data
        df = pd.read_csv(data_path)
        #df = df.iloc[index, :]
        # Pass it to a tensor through numpy and torch
        self.X = torch.from_numpy(df[self.features].values)
        self.y = torch.from_numpy(df[self.target].values)
        #self.data = torch.from_numpy(df.values)
        
        return self.X, self.y

    def __len__(self):
        return len(self.files)


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


class TimeCosSin(TransformerMixin):
     #Return self nothing else to do here
     
    @property
    def date_time(self):
        return pd.to_datetime(self.X.pop('timestamp'), format='%Y-%m-%d %H:%M:%S') 
            
    @property 
    def timestamps(self):
        return self.date_time.map(datetime.datetime.timestamp)
        
    @property
    def feature_names(self):
        return ['Day sin', 'Day cos', 'Year sin', 'Year cos']
        
    def fit( self, X, y = None  ):
        return self
    
    def transform(self, X, y=None):
        self.X = X
        day = 24*60*60 
        year = (365.2425)*day
        
        timestamp_s = self.timestamps
        X['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        X['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        X['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        X['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        
        return X.values
        
    def get_feature_names(self, input_features=None):
        return self.feature_names
        

# def bejing_pipeline():
#     num_features = ["PM2.5", 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
#     time_cols = ["timestamp"]
#     cat_features = ["wd", "station"]
#     numeric_transformer = Pipeline(steps=[
#             ("xscaler", StandardScaler())
#         ])
#     y_transformer = Pipeline(steps=[
#             ("yscaler", StandardScaler())
#         ])
#     time_transformer = Pipeline(steps=[
#             ('cos_sin', TimeCosSin())
#         ])
#     cat_transformer = OneHotEncoder(handle_unknown='ignore')

#     data_t = ColumnTransformer(
#         transformers=[                
#                 ("y", y_transformer, [num_features.pop(0)]),
#                 ("X", numeric_transformer, num_features),
#                 ("time", time_transformer, time_cols),
#                 ("cat", cat_transformer, cat_features),
                
#             ]
#         )
#     return data_t