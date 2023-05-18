from pandas import DataFrame, Series
import pandas as pd
import numpy as np


class Evaluation:
    def get_return(cls, nav_s: Series) -> Series:
        return_p = (nav_s / nav_s.shift(1) - 1).dropna()
        return return_p

    @classmethod
    def get_sharpe_ratio(cls, nav_s: Series, period_num: int):
        '''
        get annualized sharpe ratio
        '''

        return_p = cls.get_return(cls, nav_s)
        sharpe_ratio = return_p.mean() / return_p.std() * np.sqrt(period_num)
        return sharpe_ratio

    @classmethod
    def get_annualized_return(cls, nav_s: Series, period_num: int):
        '''
        get annualized sharpe ratio
        '''
        return_p = cls.get_return(cls, nav_s)
        a_r = return_p.mean() * period_num
        return a_r

    @classmethod
    def get_max_drawdown(cls, nav_s: Series):
        cummax = nav_s.expanding().max()
        drawdown = (cummax - nav_s) /cummax
        max_drawdown = max(drawdown)

        return max_drawdown

    @classmethod
    def trading_times(cls, signal: Series):
        times = len(signal[(signal == 1) & (signal.shift(1) == 0)])
        return times

    @classmethod
    def get_win_ratio(cls, signal: Series, nav_s: Series):
        point = signal[((signal == 1) & (signal.shift(-1) == 0)) | ((signal == 0) & (signal.shift(-1) == 1))]
        if len(point) % 2 != 0:
            point = point.iloc[:-1]  # drop the last one because it doesn't leave the market
        return_every = nav_s[point[point == 1].index].values / nav_s[point[point == 0].index].values - 1
        win_ratio = len(return_every[return_every > 0]) / len(return_every)
        return win_ratio

    @classmethod
    def win_loss_ratio(cls, signal: Series, nav_s: Series):
        point = signal[((signal == 1) & (signal.shift(-1) == 0)) | ((signal == 0) & (signal.shift(-1) == 1))]
        if len(point) % 2 != 0:
            point = point.iloc[:-1]  # drop the last one because it doesn't leave the market
        return_every = nav_s[point[point == 1].index].values / nav_s[point[point == 0].index].values - 1
        win_loss_ratio = return_every[return_every > 0].mean() / abs(return_every[return_every < 0].mean())
        return win_loss_ratio
