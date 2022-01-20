import pandas as pd


def preprocess(df_train):
    df_train['datetime'] = pd.to_datetime(df_train['timestamp'], unit='s')
    df_train = df_train[df_train['datetime'] < '2021-06-13 00:00:00']
    return df_train
