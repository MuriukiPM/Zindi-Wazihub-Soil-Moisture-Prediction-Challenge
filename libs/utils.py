import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import datetime

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class IndexSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Return indices of a data frame for use in other estimators."""
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        indices = df.index
        return indices.values.astype(np.float64).reshape(-1, 1)

def ts_train_test_split(df, cutoff, target):
    """Perform a train/test split on a data frame based on a cutoff date.
    Parameters
    ----------
    df: pandas dataframe 
    cutoff: date/datetime index
    target: target column

    Returns
    -------
    df_train, df_test, y_train, y_test
    """
    
    ind = df.index < cutoff
    
    df_train = df.loc[ind]
    df_test = df.loc[~ind]
    y_train = df.loc[ind, target]
    y_test = df.loc[~ind, target]
    
    return df_train, df_test, y_train, y_test

def plot_results(df,target, y_pred, residuals):
    """Plot predicted results and residuals."""
    
    ax = plt.subplot();
    to_plot = df[target]
    to_plot.plot(ax=ax);
    pd.DataFrame(y_pred,index=df.index).plot(ax=ax, color='red', marker=None, linestyle='solid',linewidth=2,markersize=None);
    # plt.xlabel('year')
    plt.ylabel(target)
    plt.legend(['true', 'predicted']);
    plt.show();

    pd.DataFrame(residuals,index=df.index).plot()
    # plt.xlabel('year')
    plt.ylabel('residuals')

class ExternalComponents(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        """Create features based on weather."""
        self.cols = cols
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        Xt = df[self.cols].fillna(method='ffill').copy() # copy() to prevent SettingWithCopyWarning
        Xt = Xt[self.cols].fillna(method='bfill').copy() # copy() to prevent SettingWithCopyWarning
        
        return Xt

class FourierComponents(BaseEstimator, TransformerMixin):
    def __init__(self, freqs):
        """Create features based on sin(2*pi*f*t) and cos(2*pi*f*t)."""
        self.freqs = freqs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.astype(np.float64)
        Xt = np.zeros((X.shape[0], 2*len(self.freqs)))
        
        for i, f in enumerate(self.freqs):

            Xt[:, 2*i]= np.cos(2*np.pi*f*X).reshape(-1) #even cols
            Xt[:, 2*i + 1] = np.sin(2*np.pi*f*X).reshape(-1) #odd cols
    
        return Xt

class ResidualFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,cols,residuals_col='residuals',window=100):
        """Generate features based on window statistics of past noise/residuals as well as other externals."""
        self.window = window
        self.cols = cols
        self.residuals_col = residuals_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame()
        df['residual'] = X[self.residuals_col] #grab the current residuals from time series fourier prediction
        df['prior'] = df['residual'].shift(1) #series with all prev residual value for each row
        df['mean'] = df['residual'].rolling(window=self.window).mean() #mean of previous 100 residuals
        df['diff'] = df['residual'].diff().rolling(window=self.window).mean() #differential of previous 100 residuals
        df = df.fillna(method='bfill') #first row endsup with Nans due to shift.
        #add to dataframe all external feats. if all: requires both forwardfill and backfill
        df[self.cols] = X[self.cols].fillna(method='bfill') 
        df[self.cols] = X[self.cols].fillna(method='ffill') 
        
        return df

class FullModel(BaseEstimator, RegressorMixin):
    def __init__(self, baseline, residual_model, steps,residuals_col='residuals'):
        """Combine a baseline and residual model to predict any number of steps in the future."""
        
        self.baseline = baseline
        self.residual_model = residual_model
        self.steps = steps
        self.residuals_col = residuals_col
        
    def fit(self, X, y):
        self.baseline.fit(X, y)
        self.resd = y - self.baseline.predict(X)
        # given current residuals, can we predict what will be the residuals n steps in the future?
        # train using all current training residuals save for last bunch
        # validating using residual values n steps in the future for each row, with NaNs at the end removed to match shape
        resd_true = self.resd.shift(-self.steps).dropna()
        Xt = X.copy()
        Xt[self.residuals_col] = self.resd #create a temporary residuals col
        self.train_data = Xt #save for predicting
        self.residual_model.fit(Xt.iloc[:-self.steps], resd_true)
        self.index_list = []
        #create index list for future time steps
        for ix in range(self.steps): 
            if ix == 0: self.index_list.append(Xt.index[-1] + datetime.timedelta(minutes=5))
            else: self.index_list.append(self.index_list[ix-1] + datetime.timedelta(minutes=5))
                
        return self
    
    def predict(self, X):
        # given a time series, use baseline to get predictions
        y_b = pd.Series(self.baseline.predict(X),  index=X.index)
        # get residuals for the time series using the residual model, shifted appropriately to match step
        resd_pred = pd.Series(self.residual_model.predict(self.train_data))
        resd_pred = resd_pred.shift(self.steps).dropna()[-self.steps:]
        resd_pred = pd.Series(resd_pred.values, index=self.index_list)
        y_pred = y_b + resd_pred
        
        return y_pred
