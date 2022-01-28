from Utils.data_load_100hz import *
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
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate
import numpy as np

np.random.seed(0)

class Main_code():
    def __init__(self):
        self.test_undersampling = True
        self.model_name = 'RF' # 'DT' / 'RF'
        self.select_feature_mode = True
        self.save_model_name = 'best_' + self.model_name

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
        
        ## delete label from x data ##
        train_data_100hz = np.delete(train_data_100hz, -1, axis=1)
        
        ## delete mu change and unknown ##
        sampled_x, sampled_y = self.split.sampling_100hz(train_data_100hz, train_labels)
        
        unique_label = np.unique(sampled_y)

        ## train/test set random split ##
        train_x, test_x, train_y, test_y = self.split.tvt_split_data(sampled_x, sampled_y)
        
        train_y = train_y.flatten()
        print("-"*20+"train_y"+"-"*20)
        self.count_and_plot(train_y)
        non_scalered_train_data_100hz = train_x
        train_labelencoded_label, train_onehotencoded_label = self.pre.one_hot_encoding(train_y)

        ## undersampling test data for same ratio ##
        test_x_resampled, test_y_resampled = RandomUnderSampler(random_state=0).fit_resample(test_x, test_y)
        print("-"*20+"train_y"+"-"*20)
        self.count_and_plot(test_y_resampled)
        non_scalered_test_data_100hz = test_x_resampled
        test_labelencoded_label, test_onehotencoded_label = self.pre.one_hot_encoding(test_y_resampled)

        tr_hz = non_scalered_train_data_100hz
        te_hz = non_scalered_test_data_100hz
            

        return tr_hz, te_hz, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list
        
    def select_feature(self):
        tr_data, te_data, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list = self.data_set()
        tr_feature = tr_data[:,[0,3,4,5,7]]
        te_feature = te_data[:,[0,3,4,5,7]]
        print("train_x.shape = ", tr_feature.shape)
        print("test_x.shape = ", te_feature.shape)
        print("train_y.shape = ", train_labelencoded_label.shape)
        print("test_y.shape = ", test_labelencoded_label.shape)
        return tr_feature, te_feature, train_labelencoded_label, test_labelencoded_label, unique_label, feature_list
        
    def train(self):
    	if self.select_feature_mode == True:
    	    train_hz, test_hz, y, test_y, unique_label, feature_list= self.select_feature()
    	elif self.select_feature_mode == False:
    	    train_hz, test_hz, y, test_y, unique_label, feature_list= self.data_set()
    	if self.model_name == 'DT':
    	    model = self.main_model.DT()
    	elif self.model_name == 'RF':
    	    model = self.main_model.RF()
    	elif self.model_name == 'SVM':
    	    model = self.main_model.SVM()
    	elif self.model_name == 'KNN':
    	    model = self.main_model.KNN()
    	
    	trained_model, time_now = self.tr_model.train_option(model)
    	print("[START Learning] " + self.model_name + " with 100hz input data")
    	history = self.tr_model.train_model(trained_model, time_now, self.save_model_name, train_hz, y)
    	score = history.score(test_hz, test_y)
    	print('accuracy : {score:.3f}'.format(score=score))
    	self.predict.predict_100hz_model(trained_model, test_hz, test_y, self.save_model_name, time_now, unique_label, score)
    	
    def data_set_for_kfold(self):
        
        ## data load ##
        train_data_100hz, train_labels, feature_list = self.dl.return_hz_label()
        
        ## delete label from x data ##
        train_data_100hz = np.delete(train_data_100hz, -1, axis=1)
        
        ## delete mu change and unknown ##
        sampled_x, sampled_y = self.split.sampling_100hz(train_data_100hz, train_labels)
        dt_clf = self.main_model.RF()
        score = cross_val_score(dt_clf, sampled_x, sampled_y, scoring='accuracy', cv=5)
        print("each accuracy : ", np.round(score, 6))
        print("mean accuracy : ", np.round(np.mean(score), 6))
    	
    def main(self):
        self.data_set_for_kfold()
        #self.train()
        

if __name__ == "__main__":   
    main_code = Main_code()
    main_code.main()
  
    
    
