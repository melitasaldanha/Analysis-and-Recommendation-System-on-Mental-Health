# -*- coding: utf-8 -*-
"""Correlation_all_features.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10L-fI9bu8YgsWOekE-kbfLbkifZlC3R5
"""

# google drive import 
from google.colab import drive
drive.mount('/content/gdrive')

!ls

cd 'gdrive/My Drive/BDA/final_project'

# Library imports 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
import glob

# View depression questionaire data 
depression_df = pd.read_csv('mental_health_depression.csv')
depression_df.head()

# Linear Regression for individual columns 
def calculate_p_value(t_stat, dof): 
    return stats.t.sf(np.abs(t_stat), dof) * 2

# Exploring the correlation betweeb every column in each of the sub datasets and each of the target columns 
def linear_regression(feature_col, target_col, feature_df, target_df):
    # Preprocessing data 
    col_feat = feature_df[feature_col].tolist()
    col_target = target_df[target_col].tolist()
    X = []
    y = []
    for a,b in zip(col_feat, col_target):
        if math.isnan(a) or math.isnan(b):
            continue
        else:
            #print(a,b)
            X.append(a)
            y.append(b)
    num_feats = len(col_feat)
    X = np.reshape(X,(len(X),1))
    Y = np.reshape(y,(len(y),1))
    # Mean centering the data and normalising 
    X = (X - np.mean(X)) 
    Y = (Y - np.mean(Y))
    if np.std(X, axis = 0) != 0:
        X = X / np.std(X, axis = 0)
    if np.std(Y, axis = 0) != 0:
        Y = Y / np.std(Y, axis = 0)
    # Calculating beta using matrix multiplication
    X_transpose = np.transpose(X)
    X_inverse = np.linalg.pinv(np.dot(X_transpose,X))
    B = np.dot(X_inverse,X_transpose)
    betas = np.dot(B,Y)
    r = np.sum(np.power((Y - np.dot(X,betas)),2))
    dof = num_feats - 2
    s_squared = r / dof
    X_meanadj_sq = np.power((X[:,0]),2)
    se = np.sum(X_meanadj_sq)
    beta_hat = betas[0,0]
    se_beta_hat = np.sqrt(s_squared/se)
    # Calculating P - values for hypothesis testing 
    t_statistic = beta_hat / se_beta_hat
    p_value = calculate_p_value(t_statistic, dof)
    corrected_p_val = p_value * 300 
    print(corrected_p_val)
    return beta_hat

print(beta_hat)

csv_file_list = glob.glob('preprocessed_datasets/*.csv')
csv_file_list.remove('preprocessed_datasets/mental_health_depression.csv')
print(csv_file_list)

df_temp = pd.read_csv(csv_file_list[0])
df_temp.head()

# Iterating over selected files in our dataset and observing correlation 
for csv_name in csv_file_list:
    print(csv_name)
    df_temp = pd.read_csv(csv_name)
    feature_cols = list(df_temp.columns.values)
    print(feature_cols)
    feature_cols.remove('SEQN')
    df_merge = pd.merge(depression_df, df_temp,  how='left', on=['SEQN'])
    df_merge = df_merge.drop('SEQN', 1)

    beta_all_targets = []
    for t_c in target_cols:
        beta_list = []
        for feature in feature_cols:
            b = linear_regression(feature, t_c, df_merge, df_merge)
            beta_list.append(b)
        beta_all_targets.append(beta_list)

    beta_mat = np.matrix(beta_all_targets)
    # Saving results to a directory 
    df_save = pd.DataFrame(beta_mat, columns=feature_cols)
    t = csv_name.split('/')[1]
    df_save.to_csv('beta_matrix_individual_feats/beta_'+ t)