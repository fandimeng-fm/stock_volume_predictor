import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import holidays
pd.options.mode.chained_assignment = None  # default='warn'

def DailyModel(daily_data):
    '''Calls all daily models, returns dictionary'''
    estimated_daily = {}
    estimated_daily['Daily'] = Model1(daily_data)
    estimated_daily['Open_Auction'], estimated_daily['Close_Auction'] = Model3(daily_data)

    return estimated_daily

def Model1(daily_data):
    ''' put it all together (make sure the date you want is within the testing set'''
    modified_data = prep_data(daily_data)
    train, test = split_dataset(modified_data, 0.7)
    res = minimize(my_objective, np.array([0.5, 0.5], dtype=float), args=train['y_true'].values)
    phi, theta = res.x
    y_hat = estimate_y(test['y_true'].values, phi, theta, True)
    total_vol_hat = round(np.exp(y_hat + test.iloc[-1]['log_20days_AM']))

    return total_vol_hat

def prep_data(daily_data):
    '''
    Dataset preparation

    Input:
        csv_file (csv str) : your stock dataset
        stock (str) : your chosen stock capitalized symbol

    Output:
         dataset
    '''
    daily_data['log_20days_AM'] = (np.log(daily_data['total_vol_m'])).rolling(20).mean()
    daily_data['y_true'] = np.log(daily_data['total_vol_m']) - daily_data['log_20days_AM'].shift()
    data = daily_data[20:]
    data['log_total_vol_m'] = np.log(data['total_vol_m'])
    data.set_index('DATE', inplace=True)

    return data

def split_dataset(dataset, split_level):
    '''Split dataset into training set and testing set.'''
    n = int(len(dataset) * split_level)
    train = dataset[:n]
    test = dataset[n:]
    train.loc[:,'y_hat'] = estimate_y(train['y_true'].values, 0.7, -0.3, False)
    return train, test

# find y_hat given y_true
# with unknown eps, phi, theta, we need to calculate them recursively
# here assume the initial eps, which represents the error is zero
def estimate_y(y_true, phi, theta, predict):
    '''
    Calculate y_hat based on known y_true

    Input:
        y_true (array): log(Vt) - 20-day moving average
        phi(float): universal standard is about 0.7
        theta(float): universal standard is about -0.3

    Outputs:
        y_hat (array):estimated y_true
    '''
    n = len(y_true)
    y_hat = np.zeros(n)
    eps = np.zeros(n)
    for t in range(1, n):
        y_hat[t] = phi * y_true[t - 1] + theta * eps[t - 1]
        eps[t] = y_true[t] - y_hat[t]

    if predict:
        predict_y = phi * y_true[-1] + theta * eps[-1]
        return predict_y
    else:
        return y_hat

def my_objective(param, y_true):
    '''
    Objective loss function for minimizing
    Find the Weighted Asymmetrical Logarithmic Error

    Input:
        param(float array): potential phi and theta
        y_true (array): log(Vt) - 20-day moving average

    Outputs:
        ALE, weighted asymmetrical logarithmic error
    '''
    phi = param[0]
    theta = param[1]
    y_hat = estimate_y(y_true, phi, theta, False)
    y_diff = y_hat - y_true
    ALE = np.sum((1.5 + 0.5 * np.sign(y_diff)) * np.abs(y_diff))
    return ALE

def Model3(daily_data):

    daily_data = exclude_option_expiration(daily_data)
    df = daily_data[-20:]

    df['Open_20D_Avg'] = np.log(df['OSize']).rolling(20).mean()
    df['Close_20D_Avg'] = np.log(df['CSize']).rolling(20).mean()

    open_vol = round(np.exp(df.iloc[-1]['Open_20D_Avg']))
    close_vol = round(np.exp(df.iloc[-1]['Close_20D_Avg']))

    return open_vol, close_vol

def exclude_option_expiration(daily_data):
    '''Remove option expiration dates from historical data'''
    start = daily_data['DATE'].dt.date.iloc[0]
    end = daily_data['DATE'].dt.date.iloc[-1]

    fridays = pd.date_range(start=start, end=end, freq='W-FRI').to_series()
    third_fridays = fridays.groupby([fridays.index.year, fridays.index.month]).nth(2)
    holiday_exception = [(fri -timedelta(days=1)) for fri in third_fridays if fri in holidays.Canada()] #US Holidays does not include Good Friday

    daily_data = daily_data.loc[~daily_data['DATE'].dt.date.isin(third_fridays)]
    daily_data = daily_data.loc[~daily_data['DATE'].dt.date.isin(holiday_exception)]

    return daily_data
