import numpy as np
from scipy.fftpack import fft


class Fft_code():
    def __init__(self):
        self.last_xlist = []
        self.last_ylist = []


    def select_feature(self, x_data):
        tr_data = x_data #(n,200,8)
        # Speed_, Lateral_ACC, Longitudinal_ACC, Yaw_, EngineTq, EngineSpeed, MPress, Steer_
        #   0         1               2           3       4           5         6       7
        feature = np.concatenate((tr_data[:,:,0:1], tr_data[:,:,4:5]), axis=2)
        feature = np.concatenate((feature, tr_data[:,:,5:6]), axis=2)
        feature = np.concatenate((feature, tr_data[:,:,7:]), axis=2)
        return feature
        
    def sampletest(self, x_data):
        all_input = []
        for i in x_data:
            sub_input = []
            for f in range(0, x_data.shape[-1], 1):
                feature = i[:, f]
                Ts = 0.01 
                Fs = 1/Ts
                t = np.arange(0,2,0.01)
                y = feature
                n = len(y) 
                k = np.arange(n)
                T = n/Fs 
                freq = k/T 
                freq = freq[range(int(n/2))] 	
                Y = np.fft.fft(y)/n 
                Y = Y[range(int(n/2))]
                sub_input.append(abs(Y))
            all_input.append(np.transpose(np.array(sub_input)))
        all_input = np.array(all_input)
        all_input = all_input.reshape(-1,all_input.shape[1]* all_input.shape[-1])
        return all_input
    
          
if __name__ == "__main__":   
    main_code = Main_code()
    main_code.sampletest()
    
