import pandas as pd
import numpy as np
from pandas import DataFrame
import glob
from Utils.count_label import *


class Data_load:
    def __init__(self):
        self.hz_data_list = glob.glob('/home/milab12/Autoterrain/AutoTerrain/use_data_3/Edited_db/db/**/*/*0Hz.csv', recursive=True)
        self.hz_data_list.sort()

    def hz_load(self, hz_data_list, overlap):
        labels = np.empty((1, 1), dtype=object)
        global feature_list
        hz_samples = np.empty((1, 9), dtype=object)
        concat_200 = []
        for i in (hz_data_list):
            data = pd.read_csv(i)       # Pandas를 활용하여 csv 파일 불러오기
            
            df = DataFrame(data)        # 불러온 csv 파일을 DataFrame으로 변환
            
            # delete nan data #
            df.dropna(inplace=True)
                
            # delete not using column #
            if 'Unnamed: 0' in df.columns:
                df = df.drop(['Unnamed: 0'], axis=1)
            df = df.drop(['Time_Sec_100Hz'], axis=1) 
             
            # delete not using column - brk_flag #
            df = df.drop(['BrkFlag'], axis = 1)

            feature_list = list(df.columns)[:-1]
            
            # delete speed<max speed/2 #
            speed_cut = int(max(list(df['Speed_']))/10)*10
            min_speed_list = df[df['Speed_'] < 10].index
            
            # combine split L/R data #
            df = df.replace('Left = Asphalt, Right = Ice', 'Split')
            df = df.replace('Left = Asphalt, Right = Snow', 'Split')
            df = df.replace('Left = Ice, Right = Asphalt', 'Split')
            df = df.replace('Left = Ice, Right = Snow', 'Split')
            
            
            df_numpy = df.to_numpy()
            
            for n in range(0,len(df_numpy),overlap):#,v in enumerate(df_numpy):                
                if (n<=len(df_numpy)-200):
                    concat_200.append(df_numpy[n:n+200])  

        to_delete_indexes = []
        for index, v in enumerate(concat_200):
            l = [ind[-1] for ind in v]
            u = np.unique(l)
            if len(u) > 1:
                to_delete_indexes.append(index)
        index_list = [i for i,v in enumerate(concat_200)]
        sub = [x for x in index_list if x not in to_delete_indexes]
        concat_200 = np.array(concat_200)[sub]
        label = [np.array(i).flatten().tolist()[-1] for i in concat_200]

        concat_200 = np.array(concat_200)
        result = concat_200.reshape(-1, concat_200.shape[-1])
        result = np.delete(result, -1, axis=1)
        result = result.reshape(-1, concat_200.shape[1],concat_200.shape[-1]-1)
        return result, label, feature_list

    def return_hz_label(self, overlap):
        data, label, feature_list = self.hz_load(self.hz_data_list, overlap)#r
        count = Count_label()
        print("*"*10+"data for ML(before delete Mu change and unknown class)"+"*"*10)
        count.new_label_check(np.array(label).reshape(-1,1))
        return data, label, feature_list

        
