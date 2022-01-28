import numpy as np



class Count_label():
    def __init__(self):
        pass

    def new_label_check(self, label_name):
        print("[start] Count Label for each file ")
        label_name = label_name.flatten()
        unique_label_name = np.unique(label_name)
        matrix = [[0 for col in range(0)] for row in range(len(unique_label_name))]
        for i in label_name:
            if i in unique_label_name:
                loc = list(unique_label_name).index(i)
                matrix[loc].append(i)

        print("*"*60)
        print("matrix[0] = " + "class = " + str(unique_label_name[0]) + " : ", len(matrix[0]))
        print("matrix[1] = " + "class = " + str(unique_label_name[1]) + " : ", len(matrix[1]))
        print("matrix[2] = " + "class = " + str(unique_label_name[2]) + " : ", len(matrix[2]))
        print("matrix[3] = " + "class = " + str(unique_label_name[3]) + " : ", len(matrix[3]))
        print("matrix[4] = " + "class = " + str(unique_label_name[4]) + " : ", len(matrix[4]))
        print("matrix[5] = " + "class = " + str(unique_label_name[5]) + " : ", len(matrix[5]))
        print("matrix[6] = " + "class = " + str(unique_label_name[6]) + " : ", len(matrix[6]))
        print("matrix[7] = " + "class = " + str(unique_label_name[7]) + " : ", len(matrix[7]))
        print("matrix[8] = " + "class = " + str(unique_label_name[8]) + " : ", len(matrix[8]))
        print("matrix[9] = " + "class = " + str(unique_label_name[9]) + " : ", len(matrix[9]))
        print("matrix[10] = " + "class = " + str(unique_label_name[10]) + " : ", len(matrix[10]))
        print("matrix[11] = " + "class = " + str(unique_label_name[11]) + " : ", len(matrix[11]))
        print("matrix[12] = " + "class = " + str(unique_label_name[12]) + " : ", len(matrix[12]))
        print("all_data = ", len(matrix[0])+len(matrix[1])+len(matrix[2])+len(matrix[3])+len(matrix[4])+len(matrix[5])+len(matrix[6])+len(matrix[7])+len(matrix[8])+len(matrix[9])+len(matrix[10])+len(matrix[11])+len(matrix[12]))
        print("*"*60) 
        
