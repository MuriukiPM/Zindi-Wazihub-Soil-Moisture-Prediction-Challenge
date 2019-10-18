import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.base import BaseEstimator, TransformerMixin

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
