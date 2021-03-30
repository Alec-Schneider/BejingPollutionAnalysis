import pandas as pd
from fredapi import Fred

fred = Fred(api_key_file='./.data/api_key.txt')

def fetch_fred(data):
    """
    :param data: dict pair of data point name and the FRED series name
    :return dfs_dict: dict of DataFrames
    """
    dfs_dict = {}
    for metric, series in data.items():
        print('Fetching...', metric)
        df = fred.get_series(series)
        df = pd.DataFrame(df, columns=[metric])
        dfs_dict[metric] = df

    return dfs_dict


# def dump_pickle(item, filename):
#     """
#     Dumps anything you need by filename into a pickle file
#     """
#     pick = open(filename, 'wb') if filename.split('.')[-1] == 'pickle' else open(filename + '.pickle', 'wb')
#     pickle.dump(item, pick)
#     pick.close()


def merge_data(dfs):
    """
    Merges all DataFrames into one DataFrame.
    Creates a daily index that takes in the oldest date of all DFs and
    latest date of all DFs
    :param dfs:
    :return:
    """
    oldest = [data.head(1).index for i, data in dfs.items()]
    oldest.sort()
    latest = [data.tail(1).index for i, data in dfs.items()]
    latest.sort(reverse=True)

    # Create DatetimeIndex from oldest oldest date to latest
    # use index one of latest since Jan 2020 has not happened yet
    # Investigate Jan 2020 data point
    dateIndex = pd.date_range(oldest[0].date[0], latest[1].date[0], freq='D')

    # Create dummy main_df
    main_df = pd.DataFrame()

    for key, df in dfs.items():
        print('Merging...', key)
        df = df.reindex(dateIndex)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    return main_df


if __name__ == "__main__":
    import json
    from pathlib import Path
    file_path = Path(r"C:\Users\Alec\OneDrive\Documents\Syracuse\IST707-DataAnalytics\project\data_dict.json")
    out_path = Path(r"C:\Users\Alec\OneDrive\Documents\Syracuse\IST707-DataAnalytics\project\.data\combined_fred_data.csv")
    
    with open(file_path, "r") as f:
        data_names = json.load(f)

    data = data_names["data"]

    data_dicts = fetch_fred(data)
    combined_data = merge_data(data_dicts)
    print("Writing data to...", out_path)
    combined_data.to_csv(out_path, index=True)

