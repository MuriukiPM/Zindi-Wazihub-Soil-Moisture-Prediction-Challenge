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

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


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
    # construct and train pipeline
    time = utils.IndexSelector()
    poly = PolynomialFeatures()
    scaler = StandardScaler()
    svr = SVR(gamma='auto')
    pipe = Pipeline([('indices', time),
                     ('drift', poly),
                     ('scaler', scaler),
                     ('regressor', svr)],memory=memory)
    return pipe


#%%
def main():
    # get the data
    _, train_df_field2, _, _, _, humidity_field2, _, _ = getdata()
    humidity_field2 = humidity_field2.values.reshape(-1)
    utils.logger.info(train_df_field2.shape)
    utils.logger.info(humidity_field2.shape)

    #rmtree(cachedir)
    cachedir = mkdtemp() #creates a temporary directory

    pipe = createpipeline(cachedir)
    utils.logger.info(pipe)

    param_grid = {"drift__degree": range(2,4),
                    "regressor__degree":range(2,4),
                    "regressor__kernel":('rbf','poly','sigmoid'),
                    "regressor__C": np.linspace(1,50,10), #>0
                    "regressor__tol":np.logspace(-4,1,10),
                    "regressor__epsilon":np.linspace(0.1,1.1,5)
                    }
    ts_cv = TimeSeriesSplit(5) # 5-fold forward chaining
    search = GridSearchCV(
        pipe, param_grid, cv=ts_cv, scoring = 'neg_mean_squared_error', verbose=1)
    utils.logger.info("Performing grid search...")
    search.fit(train_df_field2, humidity_field2)
    utils.logger.info("Best score: %0.3f" % search.best_score_)
    utils.logger.info("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        utils.logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))

    utils.logger.info("Saving model...")
    joblib.dump(search, 'GridSearch.pckl')


#%%
if __name__ == '__main__':
    main()

