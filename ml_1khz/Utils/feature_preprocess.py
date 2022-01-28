from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Preprocess:
    def __init__(self):
        pass
    def data_preprocess_1Khz(self, data_1Khz):
        result_1Khz, result_100hz = [], []
        for index in range(0, len(data_1Khz[:, :]), 10):
            result_1Khz.append(data_1Khz[index:index+10, :])
        return result_1Khz

    
         
    def one_hot_encoding(self, labels):
        le = LabelEncoder()
        result = []
        le.fit(labels)
        label_encoded = le.transform(labels)

        oe = OneHotEncoder(sparse = False)
        integer_encoded = label_encoded.reshape(len(label_encoded), 1)
        onehot_encoded = oe.fit_transform(integer_encoded)

        return label_encoded, onehot_encoded

        
        

