import matplotlib.pyplot as plt
import pandas as pd
from file_processing import makedir
from pandas import DataFrame, Series
import numpy as np

window_n = 20
index_code = '000300'
path = makedir("./data/")
feature = pd.read_csv(path + index_code + ".rolling " + str(window_n) + " beta.csv", index_col=0)
data = pd.read_csv(path + index_code + ".csv", index_col=['date'], encoding='gbk')
future_n = 10
image_path = makedir("./Image/")


class Description:
    def get_future_return(self, data: DataFrame, future_n: int) -> DataFrame:
        n_days_future_return = (
            (data['close'].rolling(future_n + 1).apply(lambda x: x[-1] / x[0] - 1)).shift(
                -future_n)).to_frame().dropna()
        n_days_future_return.columns = [str(future_n) + " days future return"]
        return n_days_future_return

    def get_corr(self, feature: DataFrame, n_days_future_return, start_date="2005-01-01", end_date="2016-12-31"):
        merge_data = pd.concat([feature, n_days_future_return], axis=1, join='inner')
        select_time = merge_data[(merge_data.index >= start_date) & (merge_data.index <= end_date)]
        corr = select_time.corr().iloc[0, 1]
        return corr

    def plot_distribution(self, feature, start_date="2005-01-01", end_date="2016-12-31"):
        feature_select = feature[(feature.index >= start_date) & (feature.index <= end_date)]
        plt.figure()
        plt.hist(x=feature_select, bins=100)
        plt.title(feature_select.columns[0])
        plt.savefig(image_path + feature_select.columns[0] + " plot distribution.png")

    def group_return_plot(self, feature, n_days_future_return, start_date="2005-01-01", end_date="2016-12-31"):
        merge_data = pd.concat([feature, n_days_future_return], axis=1, join='inner')
        merge_data = merge_data[(merge_data.index >= start_date) & (merge_data.index <= end_date)]
        merge_data['group'] = pd.cut(merge_data[feature.columns[0]], bins=60)
        select_data = merge_data[merge_data.group.isin(merge_data.groupby('group').count()[feature.columns[0]][
                                                           merge_data.groupby('group').count()['beta'] >= 5].index)]
        group_return = select_data.groupby('group').apply(lambda x: x[n_days_future_return.columns[0]].mean()).dropna()
        plt.figure()
        group_return.plot.bar()
        plt.title(feature.columns[0] + " group return.png")
        plt.savefig(image_path + feature.columns[0] + " group return.png")


start_date = "2005-01-01"
end_date = "2016-12-31"
des = Description()
n_days_future_return = des.get_future_return(data, future_n)
des.plot_distribution(feature, start_date=start_date, end_date=end_date)
corr = des.get_corr(feature, n_days_future_return, start_date=start_date, end_date=end_date)
des.group_return_plot(feature, n_days_future_return, start_date=start_date, end_date=end_date)
