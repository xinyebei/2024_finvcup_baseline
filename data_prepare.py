# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data = []
    with open("./data/finvcup9th_1st_ds5/train_label.txt", "r") as f:
        for tmp in f.readlines():
            data.append(tmp.replace("\n", "").split(","))        
    df_data = pd.DataFrame(data, columns=["wav_path", "label"])    
    df_data["wav_path"] = df_data["wav_path"].apply(lambda x:"./data/finvcup9th_1st_ds5/train/"+x)
    
    #训练集，验证集划分
    paths = df_data["wav_path"].values
    labels = df_data["label"].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(paths, labels, test_size=0.2, random_state=42)
    df_train = pd.DataFrame(X_train, columns=["wav_path"])
    df_train["label"] = y_train
    
    df_valid = pd.DataFrame(X_valid, columns=["wav_path"])
    df_valid["label"] = y_valid
    
    df_train.to_csv("./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_train_data.csv", index=False)
    df_valid.to_csv("./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_valid_data.csv", index=False)
    
    #测试集生成
    test_speeches = glob.glob(os.path.join("./data/finvcup9th_1st_ds5/test", "*.wav"))
    df_test = pd.DataFrame(test_speeches, columns=["wav_path"])
    df_test.to_csv("./data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_test_data.csv", index=False)
    
    print("done!")

if __name__=='__main__':
    main()
    