from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Main_model():
    def __init__(self):
        pass

    def RF(self):
    	classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', n_jobs =10, random_state=1, class_weight ='balanced')
    	return classifier
    	# RandomForestClassifier(n_estimators = int, default = 100
#    				criterion = 'gini' or 'entropy', default = 'gini'
#				max_depth = int, default = None
#				min_samples_split = int or float, default = 2
#				

    	
    def DT(self):
    	classifier = DecisionTreeClassifier(random_state=1, class_weight = 'balanced')
    	return classifier	    
