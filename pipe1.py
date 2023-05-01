#!/usr/bin/env python
# coding: utf-8

# # Train
# 
# #### This script trains the data

# In[1]:


# Import libraries
import os
import argparse
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from azureml.core import Run, Model
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle


# In[1]:


# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--prepped-data", type=str, dest='prepped_data')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

pickle.dump([1,2,3], open('./outputs/123_1.pkl','wb'))
run.log("hello", 1)

run.complete()

