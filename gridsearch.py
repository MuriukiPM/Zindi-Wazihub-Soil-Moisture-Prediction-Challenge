# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../../../tmp'))
	print(os.getcwd())
except:
	pass

#%%
#!/usr/bin/python3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from libs import utils

from tempfile import mkdtemp
from shutil import rmtree

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib

def main():

    train_df = pd.read_csv("data/Train.csv", header=0)# ignore the first row of the CSV file.
    cxt_maize_df = pd.read_csv("data/Context_Data_Maize.csv", header=0)
    cxt_peanuts_df = pd.read_csv("data/Context_Data_Peanuts.csv", header=0)

    train_df['datetime'] = pd.to_datetime(train_df['timestamp']) #to datetime
    train_df['time_of_day'] = train_df['datetime'].apply(lambda x: x.hour) #some properties
    train_df['month'] = train_df['datetime'].apply(lambda x: x.month)
    train_df['week_of_year'] = train_df['datetime'].apply(lambda x: x.weekofyear)
    train_df['date'] = train_df['datetime'].apply(lambda x: x.date()) #to datetime.date
    train_df = train_df.set_index('datetime')
    cxt_maize_df['date'] = pd.to_datetime(cxt_maize_df['Date'],format="%d-%b")
    cxt_maize_df['date'] = cxt_maize_df['date'].apply(lambda x: (x + pd.offsets.DateOffset(year=2019)).date()) #to datetime.date
    cxt_peanuts_df['date'] = pd.to_datetime(cxt_peanuts_df['Date'],format="%d-%b")
    cxt_peanuts_df['date'] = cxt_peanuts_df['date'].apply(lambda x: (x + pd.offsets.DateOffset(year=2019)).date()) #to datetime.date

    humidity_field1 = train_df[['Soil humidity 1']]
    humidity_field2 = train_df[['Soil humidity 2']]
    humidity_field3 = train_df[['Soil humidity 3']]
    humidity_field4 = train_df[['Soil humidity 4']]
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
    train_df['Soil humidity 1'] = humidity_field1
    train_df['Soil humidity 2'] = humidity_field2
    train_df['Soil humidity 3'] = humidity_field3
    train_df['Soil humidity 4'] = humidity_field4
    train_df.dropna(subset=['Soil humidity 2'], inplace=True)
    humidity_field2 = train_df[['Soil humidity 2']]

    # perform train/test split
    cutoff_field2 = '2019-05-20'
    target_field2 = 'Soil humidity 2'
    df_train, df_test, y_train, y_test = utils.ts_train_test_split(train_df, cutoff_field2, target_field2)

    #rmtree(cachedir)
    cachedir = mkdtemp() #creates a temporary directory

    cols = ['Air temperature (C)',
       'Air humidity (%)', 'Pressure (KPa)', 'Wind speed (Km/h)',
       'Wind gust (Km/h)', 'Wind direction (Deg)']

    # construct and train pipeline
    time = utils.IndexSelector()
    weather = utils.WeatherComponents(cols)
    union = FeatureUnion([
                        ('indices', time), 
                        ('weather' weather)])
    poly = PolynomialFeatures()
    scaler = StandardScaler()
    svr = SVR(gamma='auto')

    pipe1 = Pipeline([('union', union)])
    df_train = pipe1.transform(df_train)
    pipe2 = Pipeline([('drift', poly),
                    ('scaler', scaler),
                    ('regressor', svr)],memory=cachedir)
    param_grid = {"drift__degree": range(2,4),
                    "regressor__degree":range(2,4),
                    "regressor__kernel":('rbf','poly','sigmoid'),
                    "regressor__C": np.linspace(1,50,10), #>0
                    "regressor__tol":np.logspace(-4,1,10),
                    "regressor__epsilon":np.linspace(0.1,1.1,5)
                    }
    ts_cv = TimeSeriesSplit(5) # 5-fold forward chaining
    search = GridSearchCV(
        pipe2, param_grid, cv=ts_cv, scoring = 'r2', verbose=1)
    utils.logger("Performing grid search...")
    search.fit(df_train, y_train)
    utils.logger("Best score: %0.3f" % search.best_score_)
    utils.logger("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        utils.logger("\t%s: %r" % (param_name, best_parameters[param_name]))

    utils.logger.info("Saving model...")
    joblib.dump(search, 'GridSearch.pckl')
        
if __name__ == '__main__':
    main()

