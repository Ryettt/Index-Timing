import pandas as pd
from statsmodels.regression.rolling import RollingOLS, RollingRegressionResults
from sklearn.linear_model import LinearRegression
from pyfinance.ols import PandasRollingOLS
from pandas import DataFrame, Series
import numpy as np

path = "./"
index_code = '000300'
index_name = 'CSI300'

data = pd.read_csv(path + index_code + ".csv", index_col=['date'], encoding='gbk', parse_dates=True)

window_n = 20


class Indicator:
    @staticmethod
    def get_params(data: DataFrame, window_n: int) -> (DataFrame, DataFrame):
        model = PandasRollingOLS(data['high'], data['close'], window=window_n)
        beta = model.beta
        beta.columns = ['beta']
        r2 = model.rsq
        return beta, r2

    @staticmethod
    def zscore_beta(beta: DataFrame, window_m: int) -> Series:
        rolling_mean = beta['beta'].rolling(window_m).mean()
        rolling_std = beta['beta'].rolling(window_m).std()
        zscore_beta = ((beta['beta'] - rolling_mean) / rolling_std).dropna()
        return zscore_beta
    @staticmethod
    def get_QSR(zscore_beta: Series, r2: Series):
        qsr = (zscore_beta * r2).dropna().to_frame()
        qsr.columns=['QSR']
        return qsr
