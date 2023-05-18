import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
from file_processing import makedir
from evaluation import Evaluation

path = "./data/"
outcome_path = makedir("./outcome/")


class backtest:
    def get_signal(self, indicator: DataFrame, threshold: float, start_time="2005-01-01",
                   end_time="2025-01-01") -> Series:
        select_time_ind = indicator[(indicator.index >= start_time) & (indicator.index <= end_time)].iloc[:, 0]
        # enter at T+1 and get T+2 return
        signallong = ((select_time_ind >= threshold) & (select_time_ind.shift(1) < threshold)).astype(int).shift(2)
        # leave at T+1, so would get T return
        signalshort = -((select_time_ind < -threshold) & (select_time_ind.shift(1) >= -threshold)).astype(int)
        signal = signallong + signalshort
        signal[signal == 0] = np.nan
        signal = signal.fillna(method='ffill').fillna(0)
        signal[signal == -1] = 0
        signal.name = indicator.columns[0]
        return signal

    def get_nav_construct(self, return_series: Series, signal: Series) -> Series:
        construct_return = (return_series * signal).dropna()
        construct_nav = ((construct_return + 1).cumprod())
        construct_nav.name = signal.name
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

    def get_c_t_nav_compare_plot(self, return_series: Series, signal: Series):
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
        plt.title("nav_compare_for_" + construct_nav.name)
        plt.savefig(outcome_path + "true_construct_nav_compare_" + construct_nav.name)
        return construct_nav

    def get_evaluation(self, return_series: Series, signal: Series, ind_name: str, period_num=252) -> Series:
        construct_nav = self.get_c_t_nav_compare_plot(return_series, signal)
        evaluation_all = pd.DataFrame(
            index=['annualized_return', 'sharpe_ratio', 'max_drawdown', 'trading_times', 'win_ratio',
                   'mean_win_loss_ratio'], columns=[ind_name])
        evaluation_all.loc['annualized_return', ind_name] = Evaluation.get_annualized_return(construct_nav, period_num)
        evaluation_all.loc['max_drawdown', ind_name] = Evaluation.get_max_drawdown(construct_nav)
        evaluation_all.loc['sharpe_ratio', ind_name] = Evaluation.get_sharpe_ratio(construct_nav, period_num)
        evaluation_all.loc['trading_times', ind_name] = Evaluation.trading_times(signal)
        evaluation_all.loc['win_ratio', ind_name] = Evaluation.get_win_ratio(signal, construct_nav)
        evaluation_all.loc['mean_win_loss_ratio', ind_name] = Evaluation.win_loss_ratio(signal, construct_nav)
        return evaluation_all.iloc[:, 0]


window_n = 18
window_m = 600
index_code = "000300"
# indicator = pd.read_csv(path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS.csv", index_col=0)

data = pd.read_csv(path + index_code + ".csv", index_col=['date'], encoding='gbk', usecols=['date', 'close'])
return_series = data.close / (data.close.shift(1)) - 1

threshold = 0.7

start_time = '2005-01-01'
end_time = '2023-12-31'
bt = backtest()

# signal_all = pd.DataFrame()
# for i in range(4):
#     qsrc_nnor = pd.read_csv(
#         path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS_penalty_corr_" + str(
#             i) + ".no_normalized.csv", index_col=0)
#     signal = bt.get_signal(qsrc_nnor, threshold, start_time=start_time, end_time=end_time)
#     signal_all = pd.concat([signal_all, signal], axis=1)
# signal_all.columns = ["QRS_corr_no_normalized_" + str(x) for x in range(4)]
# eval_no = signal_all.apply(lambda x: bt.get_evaluation(return_series, x, x.name[0]))
# eval_no.to_csv(outcome_path + "QRS.no_normalized.csv")
#
# signal_all = pd.DataFrame()
# for i in range(4):
#     qsrc_n = pd.read_csv(
#         path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS_penalty_corr_" + str(
#             i) + ".normalized.csv", index_col=0)
#     signal = bt.get_signal(qsrc_n, threshold, start_time=start_time, end_time=end_time)
#     signal_all = pd.concat([signal_all, signal], axis=1)
# signal_all.columns = ["QRS_corr_normalized_" + str(x) for x in range(4)]
# signal_all.columns = ["QRS_corr_normalized_" + str(x) for x in range(4)]
# eval_nor = signal_all.apply(lambda x: bt.get_evaluation(return_series, x, x.name[0]))
# eval_nor.to_csv(outcome_path + "QRS.normalized.csv")


# change signal term
# indicator = pd.read_csv(path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS_no_corr.csv",
#                         index_col=0)
#
# signal = bt.get_signal(indicator, threshold, start_time=start_time, end_time=end_time)
# construct_nav = bt.get_c_t_nav_compare_plot(return_series, signal)
# evaluation_all = bt.get_evaluation(return_series, signal, signal.name)
# evaluation_all.to_csv(outcome_path + signal.name + ".csv")


# original change
indicator = pd.read_csv(path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS.csv",
                        index_col=0)

signal = bt.get_signal(indicator, threshold, start_time=start_time, end_time=end_time)
evaluation_all = bt.get_evaluation(return_series, signal, signal.name)
evaluation_all.to_csv(outcome_path + signal.name + ".csv")

# final change
indicator = pd.read_csv(path + index_code + ".rolling." + str(window_n) + "_" + str(window_m) + ".QRS_update.csv",
                        index_col=0)

signal = bt.get_signal(indicator, threshold, start_time=start_time, end_time=end_time)
signal.name = 'QSR_update'
evaluation_all = bt.get_evaluation(return_series, signal, signal.name)
evaluation_all.to_csv(outcome_path + signal.name + ".csv")
