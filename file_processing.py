import os

import pandas as pd


def makedir(dir):
    '''

    use to make the dir
    '''
    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir


class File_process:

    def get_file_name_end(self, file_name):
        name_end = ".".join(file_name.split(".")[1:])
        return name_end

    def get_name_date(self, file_name):
        date = file_name.split(".")[0]
        return date

    def get_filelist_between_dates(self, file_list, start_date, end_date):
        file_list_name_end = self.get_file_name_end(file_list[0])
        date_list = pd.DataFrame(data=list([self.get_name_date(x) for x in file_list]))
        date_list.columns = ['Date']
        select_date_list = [x[0] + "." + file_list_name_end for x in
                            date_list.loc[(date_list['Date'] >= start_date) & (date_list['Date'] <= end_date)].values]
        return select_date_list




# f_p = File_process()
# start_date = '20180101'
# end_date = '20220505'
#file_list = os.listdir(r"C:\RV\rv\\")
# #
# # file_list_name_end = f_p.get_file_name_end(file_list[0])
#
# print(f_p.get_filelist_between_dates(file_list,start_date,end_date))
