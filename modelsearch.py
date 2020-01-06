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
from data import data

from tempfile import mkdtemp
from shutil import rmtree

import joblib

from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score


#%%
def getdata():
    _, _, _, _, _, _, _, _, train_df_field1, train_df_field2, train_df_field3, train_df_field4 = data.datasets()
    humidity_field1 = train_df_field1[['Soil humidity 1']]
    humidity_field2 = train_df_field2[['Soil humidity 2']]
    humidity_field3 = train_df_field3[['Soil humidity 3']]
    humidity_field4 = train_df_field4[['Soil humidity 4']]
    humidity_field2.loc[
        '2019-05-25 07:45:00':'2019-05-31 09:20:00'
        ].loc[
            ~pd.isna(humidity_field2['Soil humidity 2']),'Soil humidity 2'] = np.nan
    humidity_field4.loc[
        '2019-05-25 07:45:00':'2019-05-31 09:20:00'
        ].loc[
            ~pd.isna(humidity_field4['Soil humidity 4']),'Soil humidity 4'] = np.nan
    humidity_field1.loc[
        '2019-03-25 22:50:00':'2019-05-31 09:20:00'
        ].loc[
            ~pd.isna(humidity_field1['Soil humidity 1']),'Soil humidity 1'] = np.nan
    humidity_field3.loc[
        '2019-04-19 20:15:00':'2019-05-31 09:20:00'
        ].loc[
            ~pd.isna(humidity_field3['Soil humidity 3']),'Soil humidity 3'] = np.nan
    #Update
    train_df_field1['Soil humidity 1'] = humidity_field1
    train_df_field2['Soil humidity 2'] = humidity_field2
    train_df_field3['Soil humidity 3'] = humidity_field3
    train_df_field4['Soil humidity 4'] = humidity_field4
    # Separate the test set for the competition
    train_df_field1.dropna(subset=['Soil humidity 1'], inplace=True)
    train_df_field2.dropna(subset=['Soil humidity 2'], inplace=True)
    train_df_field3.dropna(subset=['Soil humidity 3'], inplace=True)
    train_df_field4.dropna(subset=['Soil humidity 4'], inplace=True)
    # Incase targets are needed separately
    humidity_field1 = train_df_field1[['Soil humidity 1']]
    humidity_field2 = train_df_field2[['Soil humidity 2']]
    humidity_field3 = train_df_field3[['Soil humidity 3']]
    humidity_field4 = train_df_field4[['Soil humidity 4']]

    return train_df_field1, train_df_field2, train_df_field3, train_df_field4, humidity_field1, humidity_field2, humidity_field3, humidity_field4


#%%
def createpipeline(memory):
    time = utils.IndexSelector()
    poly = PolynomialFeatures(degree=2)
    scaler = StandardScaler()
    pipe = Pipeline([('indices', time),
                    ('drift', poly),
                    ('scaler', scaler)],memory=memory)
    return pipe


#%%
def modelsearch():
    # get the data
    _, train_df_field2, _, _, _, humidity_field2, _, _ = getdata()
    humidity_field2 = humidity_field2.values.reshape(-1)
    utils.logger.info(train_df_field2.shape)
    utils.logger.info(humidity_field2.shape)

    #rmtree(cachedir)
    cachedir = mkdtemp() #creates a temporary directory
    
    pipe = createpipeline(cachedir)
    
    utils.logger.info(pipe)

    # Evaluate different algorithms using cross-validation(cv)

    methods = []
    #methods.append(('LR', LinearRegression())) #no-good
    #methods.append(('RIDGE', Ridge(random_state=42))) #no-good
    #methods.append(('LASSO', Lasso(random_state=42))) #no-good
    #methods.append(('SGR', SGDRegressor(random_state=42))) #no-good               
    methods.append(('SVR', SVR(gamma='auto')))
    methods.append(('KNN', KNeighborsRegressor())) 
    methods.append(('MLP', MLPRegressor(random_state=42, max_iter=2000,activation="tanh", shuffle=False)))  
    methods.append(('GBR', GradientBoostingRegressor(random_state=42)))                             
    #methods.append(('CART', DecisionTreeRegressor(random_state=42)))
    #methods.append(('RFR', RandomForestRegressor(random_state=42, n_estimators=200)))
    #methods.append(('ETR', ExtraTreesRegressor(n_estimators=200, random_state=42)))
    #methods.append(('ABR', AdaBoostRegressor(n_estimators=200, random_state=42, base_estimator=RandomForestRegressor(random_state=42, max_depth=3))))
    #methods.append(('ABR.', AdaBoostRegressor(n_estimators=50, random_state=42, base_estimator=LinearRegression())))
    #methods.append(('ABR_', AdaBoostRegressor(n_estimators=50, random_state=42, base_estimator=DecisionTreeRegressor(random_state=42, max_depth=1))))
    #methods.append(('ABR__', AdaBoostRegressor(n_estimators=50, random_state=42, base_estimator=ExtraTreesClassifier(n_estimators=7,max_depth=2, random_state=42))))
    #methods.append(('BR', BaggingRegressor(n_estimators=200, random_state=42, base_estimator=RandomForestRegressor(random_state=42, max_depth=3))))
    #methods.append(('BR.', BaggingRegressor(n_estimators=50, random_state=42, base_estimator=LinearRegression())))
    #methods.append(('BR_', BaggingRegressor(n_estimators=50, random_state=42, base_estimator=DecisionTreeRegressor(random_state=42, max_depth=1))))
    #methods.append(('BR__', BaggingRegressor(n_estimators=50, random_state=42, base_estimator=ExtraTreesClassifier(n_estimators=7,max_depth=2, random_state=42))))
    #base_estimator=LogisticRegression(solver='lbfgs',random_state=42,class_weight=class_weights)
    #base_estimator=DecisionTreeClassifier(random_state=42, max_depth=5, class_weight=class_weights)
    #base_estimator=ExtraTreesClassifier(n_estimators=200,max_depth=5, random_state=42, class_weight=class_weights)

    results = []
    names = []

    for name, method in methods:
        #sKfold = model_selection.StratifiedKFold(n_splits = 2, random_state=42)	# cross-validation
        ts_cv = TimeSeriesSplit(5) # 5-fold forward chaining
        cv_results = cross_val_score(method, pipe.fit_transform(train_df_field2), humidity_field2,
            cv=ts_cv, scoring='neg_mean_squared_error', verbose=1)
        results.append(cv_results)
        names.append(name)
        utils.logger.info(name+" : "+cv_results)
    for i in range(len(names)):
        result = results[i]
        name = names[i]
        msg = "%s: %f mean (+/- %f) std" % (name, result.mean(), result.std())
        utils.logger.info(msg)      #performance of methods


#%%
if __name__ == '__main__':
    modelsearch()

