import pandas as pd
from datetime import datetime
from baseline.daily_model import *
from baseline.intraday_model import *
from dateutil.relativedelta import relativedelta

def main(intraday_file, daily_file, ticker, date):
    '''Calls all models. returns dictionary of results'''

    date = datetime.strptime(date, '%Y-%m-%d').date()

    daily_data, intraday_data, overnight_gap = clean_data(intraday_file, daily_file, ticker, date)

    estimated_daily = DailyModel(daily_data)
    estimated_intraday = IntradayModel(intraday_data, daily_data, estimated_daily, overnight_gap)

    prediction = {}
    prediction['daily'] = estimated_daily
    prediction['intraday'] = estimated_intraday

    return prediction

def clean_data(intraday_file, daily_file, ticker, date):
    '''Formats data and selects two years prior to prediction date.'''
    daily_data = pd.read_csv(daily_file, parse_dates=['DATE'])
    intraday_data = pd.read_csv(intraday_file)

    intraday_data = format_intraday(intraday_data)

    daily_data = daily_data.loc[daily_data['symbol'] == ticker]
    intraday_data = intraday_data.loc[intraday_data['symbol'] == ticker]

    overnight_gap = get_overnight_gap(daily_data, date)

    daily_data = daily_data.loc[
        (daily_data['DATE'].dt.date >= (date - relativedelta(years=2))) & (daily_data['DATE'].dt.date < date)]
    intraday_data = intraday_data.loc[
        (intraday_data['date'].dt.date >= (date - relativedelta(years=2))) & (intraday_data['date'].dt.date < date)]

    return daily_data, intraday_data, overnight_gap

def get_overnight_gap(daily_data, date):
    '''Calculate the overnight price gap for prediction date.'''
    daily_data['20d_sd'] = daily_data['CPrc'].rolling(20).std() #our definition of price volatility
    daily_data['price_diff'] = abs(daily_data['OPrc'] - daily_data['CPrc'].shift())
    daily_data['overnight_gap'] = daily_data['price_diff'] / daily_data['20d_sd']

    overnight_gap = daily_data.loc[daily_data['DATE'].dt.date == date, 'overnight_gap'].values[0]

    return overnight_gap

def format_intraday(intraday_data):
    '''Convert TAQ intraday date and time values from strings into datetime.
        Combine sym_root and sym_suffix to create symbols matching daily data.'''

    intraday_data['datetime'] = intraday_data['date'] + ' ' + intraday_data['time']
    intraday_data['datetime'] = pd.to_datetime(intraday_data['datetime'])
    intraday_data['date'] = pd.to_datetime(intraday_data['date'])

    intraday_data['symbol'] = intraday_data['sym_root'] + intraday_data['sym_suffix']
    intraday_data.loc[intraday_data['sym_suffix'].isnull(), 'symbol'] = intraday_data['sym_root']

    return intraday_data