from Utils.data_load_new_fft import *
from Utils.feature_preprocess import *
from Utils.tr_val_te_split_fft import *
from Utils.count_label import *
from Train.models import *
from Train.model_train import *
import new_fft
import pandas as pd
from Test.model_predict import *
import imblearn
#from imblearn.combine import *
from imblearn.under_sampling import *
from collections import Counter
import numpy as np


np.random.seed(0)

class Main_code():
    def __init__(self):
        self.overlap = 2
        self.model_name = 'DT' # 'DT' / 'RF'
        self.select_feature_mode = True # True / False
        self.save_model_name = 'fft_' +str(self.overlap)+ '_'+self.model_name

        self.dl = Data_load()
        self.pre = Preprocess()
        self.split = Split_data()
        self.main_model = Main_model()
        self.tr_model = Train_model()
        self.predict = Predict()
        self.count_label = Count_label()
        self.fft = new_fft.Fft_code()
    
    def count_and_plot(self, y):
        counter = Counter(y)
        for k,v in counter.items():
            print("Class = %s, n = %d (%.3f%%)" % (str(k),v,v/len(y)*100))
        
    def data_set(self):
        
        ## data load ##
        train_data_100hz, train_labels, feature_list = self.dl.return_hz_label(self.overlap)
        
        ## delete mu change and unknown ##
        sampled_x, sampled_y = self.split.sampling_100hz(train_data_100hz, train_labels)

        unique_label = np.unique(sampled_y)

        ## train/test set random split ##
        train_x, test_x, train_y, test_y = self.split.tvt_split_data(sampled_x, sampled_y)
        train_x_reshape = train_x.reshape(-1, train_x.shape[1]*train_x.shape[-1])
        test_x_reshape = test_x.reshape(-1, test_x.shape[1]*test_x.shape[-1])
            
        train_y = train_y.flatten()
        print("*"*60)
        print("train_y")
        print("*"*60)
        self.count_and_plot(train_y)
        non_scalered_train_data_100hz = train_x
        train_labelencoded_label, train_onehotencoded_label = self.pre.one_hot_encoding(train_y)
            
        ## undersampling test data for same ratio ##
        test_x_resampled, test_y_resampled = RandomUnderSampler(random_state=0).fit_resample(test_x_reshape, test_y)
        test_x_resampled = test_x_resampled.reshape(-1, test_x.shape[1], test_x.shape[-1])
        print("*"*60)
        print("test_y")
        print("*"*60)
        self.count_and_plot(test_y_resampled)
        non_scalered_test_data_100hz = test_x_resampled
        test_labelencoded_label, test_onehotencoded_label = self.pre.one_hot_encoding(test_y_resampled)

        tr_hz = non_scalered_train_data_100hz
        te_hz = non_scalered_test_data_100hz
        
        if self.select_feature_mode == True:
            fft_select_feature_train = self.fft.select_feature(tr_hz)
            fft_select_feature_test = self.fft.select_feature(te_hz)
            fft_train_x = self.fft.sampletest(fft_select_feature_train)
            fft_test_x = self.fft.sampletest(fft_select_feature_test)
            
        
        elif self.select_feature_mode == False:
            fft_train_x = self.fft.sampletest(tr_hz)
            fft_test_x = self.fft.sampletest(te_hz)
        
        return fft_train_x, fft_test_x, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list

        
    def train(self):
    	train_hz, test_hz, y, test_y, unique_label, feature_list= self.data_set()
    	print("*"*60)
    	print("train_hz.shape = ", train_hz.shape)
    	print("test_hz.shape = ", test_hz.shape)
    	print("*"*60)
    	if self.model_name == 'DT':
    	    model = self.main_model.DT()
    	elif self.model_name == 'RF':
    	    model = self.main_model.RF()
    	
    	trained_model, time_now = self.tr_model.train_option(model)
    	print("[START Learning] ")
    	history = self.tr_model.train_model(trained_model, time_now, self.save_model_name, train_hz, y)
    	score = history.score(test_hz, test_y)
    	print('accuracy : {score:.3f}'.format(score=score))
    	self.predict.predict_100hz_model(trained_model, test_hz, test_y, self.save_model_name, time_now, unique_label, score)
    	
    	
    def main(self):
        self.train()
        

if __name__ == "__main__":   
    main_code = Main_code()
    main_code.main()
  
    
    
