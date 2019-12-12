#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:32:41 2019

@author: liuyue
"""

#data processing
import pandas as pd 
import pickle
import numpy as np

    
def read_file(file):
    f = open(file,"rb")
    tmp = pickle.loads(f.read())
    return np.array(tmp)


def write_csv(data,file,header):
    pd.DataFrame(data).to_csv(file,header = header)
    
    
sync = 1
write_local_gradient = 2
calculation = 3
def merge(file,num_workers,times,storage_type):
    data = pd.read_csv(file,index_col = False, header = 0) 
    aver = [[],[],[]]
    data = data.iloc[:,1:data.shape[1]]
    col = data.shape[1]
    for i in range(col):
        aver[i%3] = aver[i%3]+data.iloc[:,i].tolist()
    result = []
    for i in range(3):
        result.append(sum(aver[i])/num_workers/60)
    csv_path = "/Users/liuyue/LambdaML/TimeRecord/MobileNet/" + storage_type + "/result/result_{}.csv".format(times)
    print(np.array(result).shape)
    write_csv(np.array(result).reshape((1,3)),csv_path,["sync","write local gradient","calculation"])
    return np.array(result)
if __name__ == '__main__':
    
    num_workers = 8
    num_file = 4
    storage_type = "redis"
    for times in range(1,num_file+1):

        file_path = "/Users/liuyue/LambdaML/TimeRecord/MobileNet/" + storage_type + "/{}/".format(times)
        for i in range(8):
            tmp_file_path = file_path + "time_{}".format(i)
            
            if i == 0:
                data = read_file(tmp_file_path)
                header = [index+str(i) for index in ["sync_","write local gradient_","calculation_"]]
            else:
                tmp = read_file(tmp_file_path)
                header = header +[index+str(i) for index in ["sync_","write local gradient_","calculation_"]]
                data = np.vstack((data,tmp))
        #print(data.shape)
        csv_path = file_path + "time.csv"
        write_csv(data.T,csv_path,header)
        if times == 1:
            result = merge(csv_path,num_workers,times,storage_type)
        else:
            result = result + merge(csv_path,num_workers,times,storage_type)
    print(result/num_file)
    
    
 