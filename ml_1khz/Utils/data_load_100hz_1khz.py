import pandas as pd
import numpy as np
from pandas import DataFrame
import glob
from Utils.count_label import *


class Data_load:
    def __init__(self):
        self.hz_data_list = glob.glob('/home/milab12/Autoterrain/AutoTerrain/use_data_3/Edited_db/db/**/*/*0Hz.csv', recursive=True)
        self.khz_data_list = glob.glob('/home/milab12/Autoterrain/AutoTerrain/use_data_3/Edited_db/db/**/*/*KHz.csv', recursive=True)
        self.hz_data_list.sort()
        self.khz_data_list.sort()

    

    def hz_load(self, hz_data_list):
        labels = np.empty((1, 1), dtype=object)
        labels_forcheck = np.empty((1, 1), dtype=object)
        global feature_list
        hz_samples = np.empty((1, 9), dtype=object)
        num = 0
        speed_list, len_per_file = [], []
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
            
            # combine split L/R data #
            df = df.replace('Left = Asphalt, Right = Ice', 'Split')
            df = df.replace('Left = Asphalt, Right = Snow', 'Split')
            df = df.replace('Left = Ice, Right = Asphalt', 'Split')
            df = df.replace('Left = Ice, Right = Snow', 'Split')
            
            label_forcheck = df['Mu Change Section']
            label_forcheck = label_forcheck.to_numpy()
            label_forcheck = np.reshape(label_forcheck, (-1,1))
            labels_forcheck = np.append(labels_forcheck,label_forcheck,axis=0)

            min_speed_list = df[df['Speed_'] < 10].index
            speed_list.append(min_speed_list)
            df = df.drop(min_speed_list)


            label = df['Mu Change Section']  

            label = label.to_numpy()            

            label = np.reshape(label, (-1, 1))  

            labels = np.append(labels, label, axis=0)
            len_per_file.append(len(df))
            sample = df.to_numpy()
            
            hz_samples = np.append(hz_samples, sample, axis=0)

        hz_samples = np.delete(hz_samples, [0,0], axis=0)
        labels_forcheck = np.delete(labels_forcheck, [0,0], axis=0)
        labels = np.delete(labels, [0,0], axis=0)
        count = Count_label()
        return hz_samples, labels, feature_list, speed_list, len_per_file
    
    def khz_load(self, khz_data_list):
        _, _, _, to_remove_speed_list, len_per_file = self.hz_load(self.hz_data_list)
        khz_samples = np.empty((1, 10, 8), dtype=object)
        
        # load each file to pandas
        for i, remove_speed, len_file in zip(khz_data_list, to_remove_speed_list, len_per_file):
            data = pd.read_csv(i)       # load csv file using pandas
            df = DataFrame(data)        # convert csv to dataframe
            
            if 'Unnamed: 0' in df.columns:
                df = df.drop(['Unnamed: 0'], axis=1)
            if 'Unnamed: 11' in df.columns:
                df = df.drop(['Unnamed: 11'], axis=1)
            df = df.drop(['Time_Sec_1KHz', 'Mu Change Section'], axis=1) 
            if len(df)%10 != 0:
                df = df.iloc[0:-(len(df)%10), :]

            if len(remove_speed) != 0 and (len(df)/10)<=remove_speed[-1]:
                to_slice = int(remove_speed[-1]-(len(df)/10)+1)
                while((len(df)/10)<=remove_speed[-1]):
                    remove_speed = remove_speed[:to_slice*(-1)]
            sample = df.to_numpy()   

            sample_reshape = np.reshape(sample, (-1,10,8))

            ## remove speed<10
            if len(remove_speed) != 0:
                removed_khz = np.delete(sample_reshape, remove_speed, axis=0)
            else:
                removed_khz = sample_reshape

            khz_samples = np.append(khz_samples, removed_khz, axis=0)    
        
        khz_samples = np.delete(khz_samples, [0,0,0], axis=0)         
        return khz_samples
    
    def return_khz(self):
        data_1khz = self.khz_load(self.khz_data_list)
        return data_1khz
        
    def return_hz_label(self):
        data, label, feature_list, speed_list, len_for_file = self.hz_load(self.hz_data_list)
        count = Count_label()
        print("*"*10+"data for ML(before delete Mu change and unknown class_100hz)"+"*"*10)
        count.new_label_check(label)
        return data, label, feature_list

        
