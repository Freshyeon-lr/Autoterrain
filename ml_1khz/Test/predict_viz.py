from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd

class Predict_viz:
    def __init__(self):
        pass
        
    def confusion_matrix(self, answer, predicted, model, save_name, now, unique_label,score):
        print("confusion_matrix")
        plt.figure(figsize = (10,10))
        label_list = ['Asphalt','Gravel','Ice','Mud',"Snow",'Split','Wet A','Wet G',"Wet M",'Wet WLF','WLF']
        cm = pd.DataFrame(confusion_matrix(answer, predicted), columns=[label_list], index=[label_list])

        ax = sns.heatmap(cm, annot=True)
        fig = ax.get_figure()

        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.title('Heatmap of Confusion Matrix')
        fig.savefig('./result/confusion/{0}_{1}_{2}_{3}_{4}_{5}_{6}_confusion.png'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, save_name,score), bbox_inches = 'tight', dpi = 200)
        plt.show()
        plt.cla()
        
        plt.figure(figsize = (10,10))
        cm2 = pd.DataFrame(confusion_matrix(answer, predicted, normalize = 'true'), columns=[label_list], index=[label_list])
        ax2 = sns.heatmap(cm2, annot=True)
        fig2 = ax2.get_figure()

        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.title('Heatmap of Confusion Matrix')
        fig2.savefig('./result/confusion/{0}_{1}_{2}_{3}_{4}_{5}_{6}_confusion_normalize.png'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, save_name, score), bbox_inches = 'tight', dpi = 200)
        plt.show()
        plt.cla()
       
