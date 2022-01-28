import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Preprocess:
    def __init__(self):
        pass

    def one_hot_encoding(self, labels):
        le = LabelEncoder()
        result = []
        le.fit(labels)
        label_encoded = le.transform(labels)

        oe = OneHotEncoder(sparse = False)
        integer_encoded = label_encoded.reshape(len(label_encoded), 1)
        onehot_encoded = oe.fit_transform(integer_encoded)

        return label_encoded, onehot_encoded

        
        

