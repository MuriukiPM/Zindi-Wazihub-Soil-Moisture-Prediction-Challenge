# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
#!$HOME/.local/share/virtualenvs/sk-eLjZDZHf/bin/python3


#%%
# import os
# path = os.getcwd() + "/Zindi_Wazihub_Soil_Moisture_Prediction_Challenge/"
# os.chdir(path)
# os.getcwd()


#%%
import pandas as pd
import numpy as np

from libs import utils

from tempfile import mkdtemp
from shutil import rmtree

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib

from sklearn.svm import SVR


#%%
def main:

    cachedir = mkdtemp() #creates a temporary directory


    utils.logger.info("Saving best model...")
    joblib.dump(search, 'ModelSearch.pckl')
        
if __name__ == '__main__':
    main()

