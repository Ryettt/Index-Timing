from pandas import DataFrame,Series
import pandas as pd
import numpy as np

class Evaluation:
    def get_return(cls, nav: DataFrame) -> Series:
        '''

        :params nav: nav from get_netvalue_time_series in class Account
        '''

        nav_s = nav.iloc[:, 0]
        return_p = (nav_s / nav_s.shift(1) - 1).dropna()
        return return_p

    @classmethod
    def get_sharpe_ratio(cls, nav: DataFrame, period_num: int):
        '''
        get annualized sharpe ratio

        :params nav: nav from get_netvalue_time_series in class Account
        '''
        return_p = cls.get_return(cls, nav)
        sharpe_ratio = return_p.mean() / return_p.std() * np.sqrt(period_num)
        return sharpe_ratio

    @classmethod
    def get_annualized_return(cls, nav: DataFrame, period_num: int):
        '''
        get annualized sharpe ratio

        :params nav: nav from get_netvalue_time_series in class Account
        '''
        return_p = cls.get_return(cls, nav)
        a_r= return_p.mean()* period_num
        return a_r


    @classmethod
    def get_max_drawdown(cls, nav: DataFrame):
        nav_s = nav.iloc[:, 0]
        cummax = nav_s.expanding().max()
        drawdown = cummax / nav_s - 1
        max_drawdown = max(drawdown)

        return max_drawdown
