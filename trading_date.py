import pandas as pd


class Trading_date:
    def days_bewteen_two_dates(self, date1: int, date2: int):
        '''
        date type is int

        :param date1: former date
        :param date2: later date
        :return:
        '''
        diff = len(trade_date[(trade_date.TradeDate >= date1) & (trade_date.TradeDate <= date2)])

        return diff

    def is_trading_date(self, date: int):
        if date in datelist:
            return True
        else:
            return False

    def get_n_days_later_td(self, date: int, n: int):
        '''

        get one trading date of n days later
        :return date(int)
        '''

        index1 = datelist.index(date)
        index2 = index1 + n
        return datelist[index2]

    def get_n_days_later_tds(self, date: int, n: int):
        '''

        get a list of n trading dates of n days later(contains date)
        :return date_list(int)
        '''

        index1 = datelist.index(date)
        index2 = index1 + n
        return datelist[index1:index2]

    def get_date_list_between_two_dates(self, start_date: str, end_date: str) -> list[str]:
        '''

        :param start_date:
        :param end_date:
        :return: datelist contains start_date and end_date
        '''
        index1 = datelist.index(int(start_date))
        index2 = datelist.index(int(end_date))
        return [str(x) for x in datelist[index1:index2 + 1]]


td_path = "D:/Data/Calendar/"
td_name = "TradeDates.csv"
trade_date = pd.read_csv(td_path + td_name)
datelist = list(trade_date['TradeDate'])
