import numpy as np
import pandas as pd
import datetime
#sklearn
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



class KalmanImputer(BaseEstimator, TransformerMixin):
    """
    Use Kalman Filtering to input missing numeric values 
    """
    def __init__(self):
        pass
    
    def fit(self):
        pass

    def transform(self):
        pass

class StringImputer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass
    
    def transform(self):
        pass


class WindVector(TransformerMixin):
    """
    Create a
    """
    @property
    def windspeed_col(self):
        return ["WSPM"]

    @property
    def winddir_col(self):
        return ["wd"]

    @property
    def feature_names(self):
        return ['Wx', 'Wy']
    
    @property
    def direction_vals(self):
        return {
        'E': 90, 
        'ENE': 65, 
        'ESE': 115,
        'N': 360, 
        'NE': 45, 
        'NNE': 25, 
        'NNW': 335, 
        'NW': 315,
        'S': 180,
        'SE': 135,
        'SSE': 155,
        'SSW': 205,
        'SW': 225,
        'W': 270, 
        'WNW': 295, 
        'WSW': 245}
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X

        # pop the wind direction column from the dataframe and map the string to degrees
        wind_degrees = self.X.pop(*self.winddir_col).map(self.direction_vals)        
        wind_radians = wind_degrees * np.pi / 180
        # Calculate the wind x and y components.
        wind_speed = self.X.pop(*self.windspeed_col)
        
        self.X['Wx'] = wind_speed*np.cos(wind_radians)
        self.X['Wy'] = wind_speed*np.sin(wind_radians)
        #print("Wind cols:", self.X.columns)
        return self.X

    def get_feature_names(self, input_features=None):
        return self.feature_names    

class WindowFeatures(TransformerMixin):
    @property
    def feature_names(self):
        return ["PM2.5", "pm2.5_12h"]
        
    def fit(self, X, y = None ):
        return self
    
    def transform(self, X, y=None):
        self.X = pd.DataFrame(X, columns=["PM2.5"])
        self.X["pm2.5_12h"] = self.X[["PM2.5"]].rolling(12).mean().fillna(method="backfill")
        #print("Window cols:", self.X.columns)
        return self.X
        
    def get_feature_names(self, input_features=None):
        return self.feature_names


class TimeCosSin(TransformerMixin):
    """
    Extract 
    """
     
    @property
    def date_time(self):
        return pd.to_datetime(self.X.pop('timestamp'), format='%Y-%m-%d %H:%M:%S') 
            
    @property 
    def timestamps(self):
        return self.date_time.map(datetime.datetime.timestamp)
        
    @property
    def feature_names(self):
        return ["Hour sin", "Hour cos", 'Day sin', 'Day cos', 'Year sin', 'Year cos']
        
    def fit( self, X, y = None  ):
        return self
    
    def transform(self, X, y=None):
        self.X = X
        hour = 60*60
        day = 24*hour 
        year = (365.2425)*day
        
        timestamp_s = self.timestamps
        self.X['Hour sin'] = np.sin(timestamp_s * (2 * np.pi / hour))
        self.X['Hour cos'] = np.cos(timestamp_s * (2 * np.pi / hour))
        self.X['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.X['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.X['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.X['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        #print("Time cols:", self.X.columns)
        
        return self.X
        
    def get_feature_names(self, input_features=None):
        return self.feature_names


def bejing_pipeline():
    """
    Create pipeline for Bejing Air Quality dataset
    """
    num_features = ["PM2.5", 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN']
    wd_features = ["wd","WSPM"]
    time_cols = ["timestamp"]
    cat_features = ["station"]
    numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
    y_transformer = Pipeline(steps=[            
            ("yscaler", StandardScaler()),
            #("window", WindowFeatures()),
        ])

    # window_transformer = Pipeline(steps=[
    #     ("window", WindowFeatures())
    # ])
    wind_transformer = Pipeline(steps=[
        ("wd", WindVector())
    ])
    time_transformer = Pipeline(steps=[
            ('cos_sin', TimeCosSin())
        ])
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    data_t = ColumnTransformer(
        transformers=[
                #("lag", window_transformer, [num_features[0]]),
                ("y", y_transformer, [num_features.pop(0)]),
                ("wd", wind_transformer, wd_features),
                ("X", numeric_transformer, num_features),
                ("time", time_transformer, time_cols),
                ("cat", cat_transformer, cat_features),
                
            ]

        )
    return data_t



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