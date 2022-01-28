import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import glob
from Utils.count_label import *


class Data_load:
    def __init__(self):
        self.hz_data_list = glob.glob('/home/milab12/Autoterrain/AutoTerrain/use_data_3/Edited_db/db/**/*/*0Hz.csv', recursive=True)
        self.hz_data_list.sort()

    def hz_load(self, hz_data_list):
        labels = np.empty((1, 1), dtype=object)
        global feature_list
        hz_samples = np.empty((1, 9), dtype=object)
        num = 0
        for i in (hz_data_list):
            num += 1
            data = pd.read_csv(i)
            
            df = DataFrame(data)
            
            # delete nan data #
            df.dropna(inplace=True)
                
            # delete not using column #
            if 'Unnamed: 0' in df.columns:
                df = df.drop(['Unnamed: 0'], axis=1)
            df = df.drop(['Time_Sec_100Hz'], axis=1) 
             
            # delete not using column - brk_flag #
            df = df.drop(['BrkFlag'], axis = 1)

            feature_list = list(df.columns)[:-1]
            
            min_speed_list = df[df['Speed_'] < 10].index
            df = df.drop(min_speed_list)
            
            # combine split L/R data #
            df = df.replace('Left = Asphalt, Right = Ice', 'Split')
            df = df.replace('Left = Asphalt, Right = Snow', 'Split')
            df = df.replace('Left = Ice, Right = Asphalt', 'Split')
            df = df.replace('Left = Ice, Right = Snow', 'Split')

            label = df['Mu Change Section'] 

            label = label.to_numpy() 

            label = np.reshape(label, (-1, 1)) 

            labels = np.append(labels, label, axis=0)
            
            sample = df.to_numpy()
            
            hz_samples = np.append(hz_samples, sample, axis=0)

        hz_samples = np.delete(hz_samples, [0,0], axis=0)

        labels = np.delete(labels, [0,0], axis=0)
        return hz_samples, labels, feature_list
        
    def return_hz_label(self):
        data, label, feature_list = self.hz_load(self.hz_data_list)
        count = Count_label()
        print("*"*10+"data for ML(before delete Mu change and unknown class)"+"*"*10)
        count.new_label_check(label)
        return data, label, feature_list

        
