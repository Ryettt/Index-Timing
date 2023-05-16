import pandas as pd
from file_processing import makedir
from pyfinance.ols import PandasRollingOLS
from pandas import DataFrame, Series
import numpy as np

path = "./data/"
index_code = '000300'
index_name = 'CSI300'


class Indicator:
    @staticmethod
    def get_params(data: DataFrame, window_n: int) -> (DataFrame, DataFrame):
        model = PandasRollingOLS(data['high'], data['close'], window=window_n)
        beta = model.beta
        beta.columns = ['beta']
        r2 = model.rsq
        return beta, r2

    @staticmethod
    def zscore_indicator(ind: Series, window_m: int):
        rolling_mean = ind.rolling(window_m).mean()
        rolling_std = ind.rolling(window_m).std()
        zscore_ind = ((ind - rolling_mean) / rolling_std).dropna()
        return zscore_ind

    @staticmethod
    def zscore_beta(beta: DataFrame, window_m: int) -> Series:
        zscore_beta = Indicator.zscore_indicator(beta['beta'], window_m)
        return zscore_beta

    @staticmethod
    def get_rolling_corr(data: DataFrame, window_n: int) -> Series:
        corr = data.loc[:, ['high', 'low']].rolling(window_n).corr().loc[(slice(None), 'high'), 'low'].droplevel(
            level=1).dropna()
        return corr

    @staticmethod
    def get_QSR(zscore_beta: Series, r2: Series) -> DataFrame:
        qsr = (zscore_beta * r2).dropna().to_frame()
        qsr.columns = ['QSR']
        return qsr

    @staticmethod
    def get_QSR_change_penalty(data: DataFrame, window_n: int, zscore_beta: Series, multiplier: int, nomalize=False):
        corr = Indicator.get_rolling_corr(data, window_n)
        corr_n = corr ** multiplier
        if nomalize:
            corr_n = corr_n / corr_n.expanding().mean()
        qsr_cp = (zscore_beta * corr_n).dropna().to_frame()
        qsr_cp.columns = ['QSR_corr_' + str(multiplier)]
        return qsr_cp

    def get_QSR_change_penalty_normalized(data: DataFrame, window_n: int, zscore_beta: Series, multiplier: int):
        corr = Indicator.get_rolling_corr(data, window_n)
        corr_n = corr ** multiplier
        qsr_cp = (zscore_beta * corr_n).dropna().to_frame()
        qsr_cp.columns = ['QSR_corr_' + str(multiplier)]
        return qsr_cp


data = pd.read_csv(path + index_code + ".csv", index_col=['date'], encoding='gbk', parse_dates=True)

window_n = 18
window_m = 600
beta, r2 = Indicator.get_params(data, window_n)
beta.to_csv(path + index_code + ".rolling " + str(window_n) + " beta.csv")
zscore_beta = Indicator.zscore_beta(beta, window_m)
zscore_beta.to_csv(path + index_code + ".rolling " + str(window_m) + " zscore_beta.csv")
qsr = Indicator.get_QSR(zscore_beta, r2)
qsr.to_csv(path + index_code + ".rolling " + str(window_n) + "_" + str(window_m) + " QSR.csv")

for i in range(4):
    qsrc_nnor = Indicator.get_QSR_change_penalty(data, window_n, zscore_beta, i, nomalize=False)
    qsrc_nnor.to_csv(
        path + index_code + ".rolling " + str(window_n) + "_" + str(window_m) + " QSR_penalty_corr_" + str(i) + ".no_normalized.csv")
    qsrc_n = Indicator.get_QSR_change_penalty(data, window_n, zscore_beta, i, nomalize=True)
    qsrc_n.to_csv(
        path + index_code + ".rolling " + str(window_n) + "_" + str(window_m) + " QSR_penalty_corr_" + str(i) + ".normalized.csv")

