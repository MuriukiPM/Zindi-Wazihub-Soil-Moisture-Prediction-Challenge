# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
#!$HOME/.local/share/virtualenvs/sk-eLjZDZHf/bin/python3


#%%
# import os
# path = os.getcwd() + "/Zindi_Wazihub_Soil_Moisture_Prediction_Challenge/data"
# os.chdir(path)
# os.getcwd()


#%%
import pandas as pd
import numpy as np


#%%
def datasets():
    # data
    train_df = pd.read_csv("data/Train.csv", header=0)# ignore the first row of the CSV file.
    cxt_maize_df = pd.read_csv("data/Context_Data_Maize.csv", header=0)
    cxt_peanuts_df = pd.read_csv("data/Context_Data_Peanuts.csv", header=0)
    subm = pd.read_csv("data/SampleSubmission.csv")
    train_df_raw = train_df.copy()
    subm_raw = subm.copy()
    cxt_maize_df_raw = cxt_maize_df.copy()
    cxt_peanuts_df_raw = cxt_peanuts_df.copy()

    #submission datasets
    subm['datetime'] = subm['ID'].apply(lambda x: x.split(' x ')[0])
    subm['field'] = subm['ID'].apply(lambda x: x.split(' x ')[1])
    subm_field1 = subm.loc[subm['field']=='Soil humidity 1']
    subm_field2 = subm.loc[subm['field']=='Soil humidity 2']
    subm_field3 = subm.loc[subm['field']=='Soil humidity 3']
    subm_field4 = subm.loc[subm['field']=='Soil humidity 4']
    subm_field1.loc['datetime'] = pd.to_datetime(subm_field1['datetime'])
    subm_field1 = subm_field1.set_index('datetime', drop=True)
    subm_field1.drop('field', axis=1, inplace=True)
    subm_field2.loc['datetime'] = pd.to_datetime(subm_field2['datetime'])
    subm_field2 = subm_field2.set_index('datetime', drop=True)
    subm_field2.drop('field', axis=1, inplace=True)
    subm_field3.loc['datetime'] = pd.to_datetime(subm_field3['datetime'])
    subm_field3 = subm_field3.set_index('datetime', drop=True)
    subm_field3.drop('field', axis=1, inplace=True)
    subm_field4.loc['datetime'] = pd.to_datetime(subm_field4['datetime'])
    subm_field4 = subm_field4.set_index('datetime', drop=True)
    subm_field4.drop('field', axis=1, inplace=True)

    #training datasets
    train_df['datetime'] = pd.to_datetime(train_df['timestamp']) #to datetime
    train_df['time_of_day'] = train_df['datetime'].apply(lambda x: x.hour) #some properties
    train_df['month'] = train_df['datetime'].apply(lambda x: x.month)
    train_df['week_of_year'] = train_df['datetime'].apply(lambda x: x.weekofyear)
    train_df['date'] = train_df['datetime'].apply(lambda x: x.date()) #to datetime.date
    # train_df = train_df.set_index('datetime')
    cxt_maize_df['date'] = pd.to_datetime(cxt_maize_df['Date'],format="%d-%b")
    cxt_maize_df['date'] = cxt_maize_df['date'].apply(lambda x: (x + pd.offsets.DateOffset(year=2019)).date()) #to datetime.date
    cxt_peanuts_df['date'] = pd.to_datetime(cxt_peanuts_df['Date'],format="%d-%b")
    cxt_peanuts_df['date'] = cxt_peanuts_df['date'].apply(lambda x: (x + pd.offsets.DateOffset(year=2019)).date()) #to datetime.date
    to_drop = ['timestamp','date','Date'] #after merge
    to_drop_field1 = ['Soil humidity 2', 'Irrigation field 2', 'Soil humidity 3', 'Irrigation field 3', 'Soil humidity 4', 'Irrigation field 4']
    train_df_field1 = train_df.drop(to_drop_field1, axis=1)
    train_df_field1 = train_df_field1.merge(cxt_maize_df, on=['date'], how='left')
    train_df_field1 = train_df_field1.set_index('datetime', drop=True)
    train_df_field1 = train_df_field1.drop(to_drop, axis=1)
    to_drop_field2 = ['Soil humidity 1', 'Irrigation field 1', 'Soil humidity 3', 'Irrigation field 3', 'Soil humidity 4', 'Irrigation field 4']
    train_df_field2 = train_df.drop(to_drop_field2, axis=1)
    train_df_field2 = train_df_field2.merge(cxt_peanuts_df, on=['date'], how='left')
    train_df_field2 = train_df_field2.set_index('datetime', drop=True)
    train_df_field2 = train_df_field2.drop(to_drop, axis=1)
    to_drop_field3 = ['Soil humidity 1', 'Irrigation field 1', 'Soil humidity 2', 'Irrigation field 2', 'Soil humidity 4', 'Irrigation field 4']
    train_df_field3 = train_df.drop(to_drop_field3, axis=1)
    train_df_field3 = train_df_field3.merge(cxt_peanuts_df, on=['date'], how='left')
    train_df_field3 = train_df_field3.set_index('datetime', drop=True)
    train_df_field3 = train_df_field3.drop(to_drop, axis=1)
    to_drop_field4 = ['Soil humidity 1', 'Irrigation field 1', 'Soil humidity 2', 'Irrigation field 2', 'Soil humidity 3', 'Irrigation field 3']
    train_df_field4 = train_df.drop(to_drop_field4, axis=1)
    train_df_field4 = train_df_field4.merge(cxt_peanuts_df, on=['date'], how='left')
    train_df_field4 = train_df_field4.set_index('datetime', drop=True)
    train_df_field4 = train_df_field4.drop(to_drop, axis=1)

    return train_df_raw, subm_raw, cxt_maize_df_raw, cxt_peanuts_df_raw, subm_field1, subm_field2, subm_field3, subm_field4, train_df_field1, train_df_field2, train_df_field3, train_df_field4

