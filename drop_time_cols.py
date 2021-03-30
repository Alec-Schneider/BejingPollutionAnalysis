"""
Create the timestamp index for each file and drop
the columns used to create it. Write the files to the 
out_dir passed to the command line.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from preprocessor import Preprocessor
from functools import reduce
import multiprocessing as mp
import os 
import glob
import click


def drop_time_cols(path, out_dir, drop_cols):
    file_name = os.path.basename(path)
    print("Processing.. {}".format(file_name))
    df = pd.read_csv(path, na_values="NA")


    df["timestamp"] = pd.to_datetime(df[drop_cols])
    df.set_index("timestamp", inplace=True)
    df.drop(columns=drop_cols, inplace=True)

    out_path = os.path.join(out_dir, file_name)
    print("Writing\n{}".format(out_path))
    df.to_csv(out_path, index=True)
    print("Finished... {}".format(file_name))


@click.option("--path", default=None, help="Enter path containing data")
@click.option("--out_dir", default=None, help="Enter path to rewrite files to")
@click.option("--drop_cols", default="True", help="list of columns to drop from dataframe")
@click.command()
def run_drop(path, out_dir, drop_cols):

    if drop_cols.lower() == "true":
        drop_cols = ['year', 'month', 'day', 'hour']
    else:
        drop_cols = None
    
    data_paths = glob.glob(os.path.join(path, "*.csv"))
    # Repeat out_dir and drop_cols for each path in data_paths
    drop_cols = [drop_cols for _ in range(len(data_paths))]
    out_dir = [out_dir for _ in range(len(data_paths))] 
    args = [*zip(data_paths, out_dir, drop_cols)]

    with mp.Pool(5) as p:
        p.starmap(drop_time_cols, args)

if __name__ == "__main__":
    run_drop()