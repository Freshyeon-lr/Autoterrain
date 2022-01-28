from . import predict_viz
import numpy as np

class Predict:
    def __init__(self):
        self.pr_viz = predict_viz.Predict_viz()
    
    def predict_100hz_model(self, model, test_x, test_y, save_name, now, unique_label, score):
        prediction = model.predict(test_x)
        print("--------------------------------prediction----------------------------")
        print(prediction)
        self.pr_viz.confusion_matrix(test_y, prediction, model, save_name, now, unique_label,score)

   
