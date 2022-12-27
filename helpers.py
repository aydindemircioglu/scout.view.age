import numpy as np
import os
import pandas as pd
import sys
import shutil


def recreatePath (path):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass
    os.makedirs (path)
    pass

 #

def getData (split, basePath = "../data"):
    df_train_file = os.path.join(basePath, split+"_final.csv")
    df_train = pd.read_csv(df_train_file)
    df_train["dummy_col"] = 1
    df_train["Age"] = df_train["age"] # copy over calculated age (=correct)
    df_train = df_train[np.isnan(df_train["Age"]) == False]
    return df_train


def get_image_list():
    # global here
    train_df = getData("train")
    val_df = getData("val")
    test_df = getData("test")
    return train_df, val_df, test_df


#
