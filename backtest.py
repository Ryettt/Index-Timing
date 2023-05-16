import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt

path = "./data/"

window_n = 18
window_m = 600
index_code = "000300"
indicator = pd.read_csv(path + index_code + ".rolling " + str(window_n) + "_" + str(window_m) + "QSR.csv", index_col=0)

data = pd.read_csv(path + index_code + ".csv", index_col=['date'], encoding='gbk', usecols=['date', 'close'])
return_series = data.close / (data.close.shift(1)) - 1

threshold = 0.7


class backtest:
    def get_signal(self, indicator: DataFrame, threshold: float, start_time="2005-01-01",
                   end_time="2025-01-01") -> Series:
        select_time_ind = indicator[(indicator.index >= start_time) & (indicator.index <= end_time)].iloc[:, 0]
        signallong = ((select_time_ind >= threshold) & (select_time_ind.shift(1) < threshold)).astype(int)
        signalshort = -((select_time_ind < -threshold) & (select_time_ind.shift(1) >= -threshold)).astype(int)
        signal = signallong + signalshort
        signal[signal == 0] = np.nan
        signal = signal.fillna(method='ffill').fillna(0)
        signal[signal == -1] = 0
        signal = (signal.shift(2)).dropna()  # because get the T+2 return
        return signal

    def get_nav_construct(self, return_series: Series, signal: Series):
        construct_return = (return_series * signal).dropna()
        construct_nav = (construct_return + 1).cumprod()
        return construct_nav

    def get_multi_signal_nav_compare(self, return_series: Series, signals: DataFrame):
        '''

        :param return_series:
        :param signals: Dataframe with every column as one signal of one indicator
        :return:
        '''
        return_df = signals.apply(lambda x: x * return_series.dropna())
        nav_df = return_df.apply(lambda x: (1 + x).cumprod())
        return nav_df

    def get_c_t_nav_compare_plot(self, return_series: Series, signal):
        '''
        get one construct nav and true nav compare plot
        :param return_series:
        :param signal:
        :return:
        '''
        construct_nav = self.get_nav_construct(return_series, signal)
        true_nav = (return_series.loc[construct_nav.index] + 1).cumprod()
        nav_compare = pd.concat([construct_nav, true_nav], axis=1)
        nav_compare.columns = ['construct', 'true']
        nav_compare.index = pd.to_datetime(nav_compare.index)
        nav_compare.plot()
        plt.legend()
        plt.title("nav compare")
        plt.savefig("true construct nav compare")
