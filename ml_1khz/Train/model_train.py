import time
import pickle
import joblib

class Train_model:
    def __init__(self):
        pass
        
    def train_option(self, model):
        now = time.localtime()
        return model, now
        
    def train_model(self, model, now, save_name, train_x, train_y):
        history = model.fit(train_x, train_y)
        model_path = './result/model/{0}_{1}_{2}_{3}_{4}_{5}_model.pkl'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, save_name)
        joblib.dump(history, model_path)
        return history
