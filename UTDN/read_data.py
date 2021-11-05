import numpy as np
import csv
from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import seaborn as sns
import statsmodels.api as sm

def read_df():
    def one_station_filled(test):
        blank_day = [-9999 for i in range(14)]
        # test = np.array(test)
        filled_data = []
        cur = 0.0
        count = 0
        dayDel_flag = 0

        for i, one_hour in enumerate(test):  # 13dim
            while cur != one_hour[-1]:
                blank_day[11] = one_hour[11]
                blank_day[12] = one_hour[12]
                blank_day[13] = cur
                filled_data.append(blank_day)
                count += 1

                cur = cur + 1
                if cur == 24:
                    cur = 0

            filled_data.append(one_hour[:14])
            cur = cur + 1
            if cur == 24:
                cur = 0
        length = len(filled_data)
        filled_data = np.array(filled_data)

        trans = filled_data.transpose((1, 0))
        # fill_blank
        result = []
        for one_type in trans:
            one_type_temp = np.array(list(one_type))
            index = np.where(one_type == -9999)
            orin_index = np.delete(np.array(range(0, length)), index[0], axis=0)
            orin_value = np.delete(one_type_temp, index[0], axis=0)

            filled = np.interp(index[0], orin_index, orin_value)
            for blank, ind in zip(filled, index[0]):
                one_type_temp[ind] = blank

            # index2 = np.where(one_type_temp > 9999)
            # if len(index2)>0:
            #     orin_index = np.delete(np.array(range(0, length)), index2[0], axis=0)
            #     orin_value = np.delete(one_type_temp, index2[0], axis=0)
            #     filled = np.interp(index2[0], orin_index, orin_value)
            #     for blank, ind in zip(filled, index[0]):
            #         one_type_temp[ind] = blank
            result.append(one_type_temp.tolist())
        # filled_data.replace(-9999., np.NaN, inplace=True)
        # result = result[:12] #11feature          +1station_id=12
        # print(np.where(result == -9999))
        result = np.array(result)
        result = result.transpose((1, 0))
        return result


    'read data and fill the missing data'
    index_str = 'PM25 PM10 NO2 CO O3 SO2 temp pressure humidity wind_dir wind_speed'
    index_list = index_str.split(' ')

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d %H:%M')
    # df = pd.read_csv('aoti2017.csv',parse_dates=['time'],date_parser=dateparse)
    df = pd.read_csv('beijing17_18.csv', parse_dates=['time'], date_parser=dateparse)
    df['new_id'] = df['station_id'].astype('category').cat.codes
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df = df.drop_duplicates(subset=['station_id', 'time'],keep='first')
    #8212
    listType = df['station_id'].unique()
    results = []
    station_indx = []
    pre = 0
    for i in range(len(listType)):
        one_station = df[df['station_id'].isin([listType[i]])]
        one_station.replace(999999, -9999, inplace=True)
        test = one_station.iloc[:, 2:].as_matrix()
        result = one_station_filled(test)
        if i>0:
            result = result[24:]
        length = len(result)
        # print(length)
        station_indx.append([pre,length+pre])
        results.append(result)
        pre = length+pre

    # cate_list = []
    # for i,station in enumerate(results):
    #     for j,line in enumerate(station):
    #         if 56.25 in line:
    #             cate_list.append([i,j])
    # a1 = results[18][6717]

    # cate_value = []
    # for index in cate_list:
    #     values = results[index[0]][index[1]]
    #     # if 60.5 in values:
    #     cate_value.append(values)
    # print(cate_value)
    return results,station_indx

