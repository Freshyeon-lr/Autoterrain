from sklearn.model_selection import train_test_split
import numpy as np
import random

class Split_data:
    def __init__(self):
        pass
        
    def sampling_100hz(self, data_100hz, label):    	
        mu_indexes = []
        mu_count = 0
        unk_indexes = []
        unk_count = 0

        for index, value in enumerate(label):
            if value == 'Mu Change':
                mu_indexes.append(index)
                mu_count += 1
            elif value == 'Unknown':
                unk_indexes.append(index)
                unk_count += 1
       
        total_indexes = mu_indexes+unk_indexes
        total_count = mu_count + unk_count
        samplelist = random.sample(total_indexes, total_count)
        samplelist.sort()
        sampled_data = np.delete(data_100hz, samplelist, axis=0)
        sampled_label = np.delete(label, samplelist, axis=0)

        sampled_hz = np.array(sampled_data)
        return sampled_hz, sampled_label, total_indexes

        
    
    def tvt_split_data(self, data, label):
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.3, random_state = 10)
        return x_train, x_test, y_train, y_test
    
