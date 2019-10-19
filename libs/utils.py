import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

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

class WeatherComponents(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        """Create features based on sin(2*pi*f*t) and cos(2*pi*f*t)."""
        self.cols = cols
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        Xt = df[self.cols].fillna(method='ffill').copy() # copy() to prevent SettingWithCopyWarning
        
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
    def __init__(self, window=100):
        """Generate features based on window statistics of past noise/residuals."""
        self.window = window
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame()
        df['residual'] = pd.Series(X, index=X.index)
        df['prior'] = df['residual'].shift(1) #series with all prev residual value for each row
        df['mean'] = df['residual'].rolling(window=self.window).mean() #mean of previous 100 residuals
        df['diff'] = df['residual'].diff().rolling(window=self.window).mean() #differential of previous 100 residuals
        df = df.fillna(method='bfill') #first row endsup with Nans due to shift.
        
        return df

class FullModel(BaseEstimator, RegressorMixin):
    def __init__(self, baseline, residual_model, steps):
        """Combine a baseline and residual model to predict any number of steps in the future."""
        
        self.baseline = baseline
        self.residual_model = residual_model
        self.steps = steps
        
    def fit(self, X, y):
        self.baseline.fit(X, y)
        self.resd = y - self.baseline.predict(X)
        # given current residuals, can we predict what will be the residuals n steps in the future?
        # train using all current training residuals save for last bunch
        # validating using residual values n steps in the future for each row, 
        # removing NaNs at the end to match shape
        self.residual_model.fit(self.resd.iloc[:-self.steps], self.resd.shift(-self.steps).dropna())
                
        return self
    
    def predict(self, X):
        y_b = pd.Series(self.baseline.predict(X))
        resd_pred = pd.Series(self.residual_model.predict(self.resd), index=self.resd.index)
        resd_pred = resd_pred.shift(self.steps).dropna()[-self.steps:]
        y_pred = y_b.values + resd_pred.values
        
        return y_pred
