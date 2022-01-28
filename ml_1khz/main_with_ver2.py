from Utils.data_load_100hz_1khz import *
from Utils.feature_preprocess import *
from Utils.tr_val_te_split import *
from Utils.count_label import *
from Train.models import *
from Train.model_train import *
import pandas as pd
from Test.model_predict import *
import imblearn
from imblearn.under_sampling import *
from collections import Counter
import numpy as np

np.random.seed(0)

class Main_code():
    def __init__(self):
        self.model_name = 'DT' # 'DT' / 'RF'
        self.save_model_name = 'data1khz_'+self.model_name
        self.select_feature_mode = False # True / False
        self.dl = Data_load()
        self.pre = Preprocess()
        self.split = Split_data()
        self.main_model = Main_model()
        self.tr_model = Train_model()
        self.predict = Predict()
        self.count_label = Count_label()
    
    def count_and_plot(self, y):
        counter = Counter(y)
        for k,v in counter.items():
            print("Class = %s, n = %d (%.3f%%)" % (str(k),v,v/len(y)*100))
        
    def data_set(self):
        
        ## data load ##
        train_data_100hz, train_labels, feature_list = self.dl.return_hz_label()
        train_data_1khz = self.dl.return_khz()
        if train_data_100hz.shape[0] != train_data_1khz.shape[0]:
            if train_data_100hz.shape[0] > train_data_1khz.shape[0]:
                diff = train_data_100hz.shape[0] - train_data_1khz.shape[0]
                train_data_100hz = train_data_100hz[:-(diff)]
            elif train_data_100hz.shape[0] < train_data_1khz.shape[0]:
                diff = train_data_1khz.shape[0] - train_data_100hz.shape[0]
                train_data_1khz = train_data_1khz[:-(diff)]
        train_data_100hz = np.delete(train_data_100hz, -1, axis=1)
        
        ## delete mu change and unknown ##
        sampled_x, sampled_y, remove_indexes = self.split.sampling_100hz(train_data_100hz, train_labels)
        removed_sampled_khz = np.delete(train_data_1khz, list(remove_indexes), axis=0)
        unique_label = np.unique(sampled_y)

        ## 1khz + 100hz data
        removed_sampled_khz = removed_sampled_khz[:,0,:]
        concat_data = np.concatenate((sampled_x, removed_sampled_khz),axis=1)

        ## train/test set random split ##
        train_x, test_x, train_y, test_y = self.split.tvt_split_data(concat_data, sampled_y)
            
        train_y = train_y.flatten()
        print("-------------------- train_y --------------------")
        self.count_and_plot(train_y)
        for_scaler_fit = train_x
        non_scalered_train_data_100hz = train_x
        train_labelencoded_label, train_onehotencoded_label = self.pre.one_hot_encoding(train_y)
            
        ## undersampling test data for same ratio ##
        test_x_resampled, test_y_resampled = RandomUnderSampler(random_state=0).fit_resample(test_x, test_y)
        print("-------------------- test_y --------------------")
        self.count_and_plot(test_y_resampled)
        non_scalered_test_data_100hz = test_x_resampled
        test_labelencoded_label, test_onehotencoded_label = self.pre.one_hot_encoding(test_y_resampled)

        tr_hz = non_scalered_train_data_100hz
        te_hz = non_scalered_test_data_100hz

        return tr_hz, te_hz, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list
        
    def select_feature(self):
        tr_data, te_data, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list = self.data_set() 
        # Speed_, Lateral_ACC, Longitudinal_ACC, Yaw_, EngineTq, EngineSpeed, MPress, Steer_        
        #   0          1              2           3       4           5          6      7
        tr_feature = tr_data[:,[0,3,4,5,7,8,9,10,11,12,13,14,15]]
        te_feature = te_data[:,[0,3,4,5,7,8,9,10,11,12,13,14,15]]
        print("train_x.shape = ", tr_feature.shape)
        print("test_x.shape = ", te_feature.shape)
        print("train_y.shape = ", train_labelencoded_label.shape)
        print("test_y.shape = ", test_labelencoded_label.shape)
        return tr_feature, te_feature, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list
        
    def train(self):
    	if self.select_feature_mode == False:
    	    train_hz, test_hz, y, test_y, unique_label, feature_list= self.data_set()
    	elif self.select_feature_mode == True:
    	    train_hz, test_hz, y, test_y, unique_label, feature_list= self.select_feature()
    	print("#### train_hz")
    	print(train_hz.shape)
    	print("#### test_hz")
    	print(test_hz.shape)
    	if self.model_name == 'DT':
    	    model = self.main_model.DT()
    	elif self.model_name == 'RF':
    	    model = self.main_model.RF()
    	
    	trained_model, time_now = self.tr_model.train_option(model)
    	print("[START Learning] " + self.model_name + " with 100hz input data")
    	history = self.tr_model.train_model(trained_model, time_now, self.save_model_name, train_hz, y)
    	score = history.score(test_hz, test_y)
    	print('accuracy : {score:.3f}'.format(score=score))
    	self.predict.predict_100hz_model(trained_model, test_hz, test_y, self.save_model_name, time_now, unique_label, score)

    def main(self):
        self.train()
        

if __name__ == "__main__":   
    main_code = Main_code()
    main_code.main()
