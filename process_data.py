import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from preprocessor import Preprocessor
from functools import reduce
import multiprocessing as mp
from preprocessor import bejing_pipeline, get_feature_names

import os 
import glob
import click


# def kalman_impute(series):
#     arr = prep.kalman_impute(series)


def fill_missing_strings(series, sample_size=200, samples=100):
    """
    Get the dsitribution of a unique value appearing in the column,
    sample the distribution randomly, and select n (where n = number of missing values in
    the array), and impute the randomly chosen values
    """
    fill = series.value_counts().index[0]
    filled_arr = series.fillna(fill)
    return filled_arr


def get_numeric_null_cols(df, null_cols):
    """
    Get the column names that are numeric types with missing values
    """
    cols = df[null_cols].columns
    num_bool = [is_numeric_dtype(df[col]) for col in cols]
    num_null_cols = [col for col, tf in zip(cols, num_bool) if tf]
    return num_null_cols


def get_string_null_cols(df, null_cols):
    """
    Get the column names that are string types with missing values
    """
    cols = df[null_cols].columns
    str_bool = [is_string_dtype(df[col]) for col in cols]
    str_null_cols = [col for col, tf in zip(cols, str_bool) if tf]
    return str_null_cols

def feature_engineering(df, pipeline=bejing_pipeline):
    print("Fitting {}".format(pipeline.steps))
    vals = pipeline.fit_transform(df)
    columns = get_feature_names(pipeline)
    new_df = pd.DataFrame(val, columns=columns)
    print("New DF ")
    return new_df



def run_imputation(df):
    """
    Fill in missing numeric values using Kalman Filtering,
    and fill in missing null 
    """
    
    # Create the dateIndex
    time_cols = ['year', 'month', 'day', 'hour']
    df["timestamp"] = pd.to_datetime(df[time_cols])
    df.set_index("timestamp", inplace=True)
    df.drop(columns=time_cols, inplace=True)
    # Check if there are null values in the dataset and get the columns
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].index.values
    numeric_null_cols = get_numeric_null_cols(df, null_cols)
    obj_null_cols = get_string_null_cols(df, null_cols)
    
    # Use Kalman Filtering to impute missing numeric
    # values
    for col in numeric_null_cols:
        prep = Preprocessor()
        arr = prep.kalman_impute(df[col])
        df[col] = arr

        # Backfill any missing data at the beginning of the array
        if df[col].isnull().sum():
            df[col].fillna(method="bfill", inplace=True)

    # Random draw based on distribution of 
    # unique vals in each column 
    for col in obj_null_cols:
        arr = fill_missing_strings(df[col])
        df[col] = arr

    return df



def process_files(path, out_dir, drop_cols):
    """
    Read in and write out the data, call the imputation funtion
    """
    file_name = os.path.basename(path)
    print("Processing.. {}".format(file_name))

    df = pd.read_csv(path, na_values="NA")
    df.drop(columns=drop_cols, inplace=True)
    print("Beginning Imputation for.. {}".format(file_name))
    df = run_imputation(df)

    out_path = os.path.join(out_dir, file_name)
    print("Writing\n{}".format(out_path))
    df.to_csv(out_path, index=True)
    print("Finished... {}".format(file_name))



@click.option("--path", default=None, help="Enter path containing data")
@click.option("--out_dir", default=None, help="Enter path to rewrite files to")
@click.option("--drop_cols", default="True", help="list of columns to drop from dataframe")
@click.command()
def run_in_parallel(path, out_dir, drop_cols):
    """
    Use command line flags to process the files in parallel
    """
    
    if drop_cols.lower() == "true":
        drop_cols = ["No"]
    else:
        drop_cols = None
    
    data_paths = glob.glob(os.path.join(path, "*.csv"))
    # Repeat out_dir and drop_cols for each path in data_paths
    drop_cols = [drop_cols for _ in range(len(data_paths))]
    out_dir = [out_dir for _ in range(len(data_paths))] 
    args = [*zip(data_paths, out_dir, drop_cols)]
    print(args[0])
    with mp.Pool(3) as p:
        p.starmap(process_files, args)



if __name__ == "__main__":
    run_in_parallel()
        



            