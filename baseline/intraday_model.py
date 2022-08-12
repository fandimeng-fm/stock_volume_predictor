import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression
import pandas_market_calendars as mcal
from scipy import stats

warnings.filterwarnings('ignore')

def IntradayModel(intraday_data, daily_data, estimated_daily, overnight_gap):
    '''Intakes intraday and daily volume data and estimated daily volume from Model 1. Returns intraday volume curve prediction.'''
    historical = agg_intraday_daily(intraday_data, daily_data)
    coef = regress_volume(historical)
    estimated_intraday = predict_intraday(coef, historical, estimated_daily, overnight_gap)

    return estimated_intraday

def agg_intraday_daily(intraday_data, daily_data):
    '''Aggregate intraday and daily data for a specific symbol.'''

    daily_data.rename(columns={'DATE':'date'}, inplace=True)

    intraday_data = eliminate_half_days(intraday_data)
    daily_data = eliminate_half_days(daily_data)

    daily_data['daily_vol_pct'] = daily_data['total_vol_m'].rank(pct=True)

    df = intraday_data.merge(daily_data[['date', 'symbol', 'total_vol_m', 'overnight_gap', 'daily_vol_pct']], how='left', on=['date', 'symbol'])
    df.index = pd.DatetimeIndex(df['datetime'])
    df['time'] = df.index.time

    df.drop(columns=['datetime', 'sym_root', 'sym_suffix'], inplace=True)

    df = df.between_time('09:31:00', '15:59:00')
    df['%_vol'] = df['size'] / df['total_vol_m']

    df.dropna(0, inplace=True)

    return df

def eliminate_half_days(data):
    '''Remove half days from historical data'''
    nyse = mcal.get_calendar('NYSE')
    date_range = nyse.schedule(start_date='2019-01-01', end_date='2021-12-31') #hardcoded - refactor to parameters
    early_close = nyse.early_closes(schedule=date_range).index
    early_close = early_close.to_series().dt.strftime('%Y%m%d')

    data = data.loc[~data['date'].dt.date.isin(early_close)]

    return data

def regress_volume(data):
    '''Regress volume data for each minute.'''
    x = data.groupby(data.index.date)[['overnight_gap', 'daily_vol_pct']].first()
    y_df = data.pivot(index = 'date', columns='time', values='%_vol')
    y_df.fillna(0, inplace=True)

    coef = []
    intercept = []
    for t in sorted(data['time'].unique()):
        y = y_df[t]

        regression_model = LinearRegression()
        regression_model.fit(x, y)

        coef.append(regression_model.coef_)
        intercept.append(regression_model.intercept_)

    coef = pd.DataFrame(coef, columns = [ 'b1' , 'b2' ])
    coef['b0'] = intercept
    coef.index = sorted(data['time'].unique())

    return coef

def predict_intraday(coef, historical, estimated_daily, overnight_gap):
    '''Predicts intraday volume for given date'''
    est_daily_vol = estimated_daily['Daily']
    vol_pct = stats.percentileofscore(historical['total_vol_m'], est_daily_vol)/100

    coef['est_%_vol'] = coef['b0'] + overnight_gap * coef['b1'] + vol_pct * coef['b2']
    coef['est_%_vol_smooth'] = coef['est_%_vol'].rolling(5).mean().shift(-2)

    coef['est_vol'] = round(coef['est_%_vol_smooth'] * est_daily_vol)
    coef.loc[coef['est_vol'].isnull(), 'est_vol'] = round(coef['est_%_vol'] * est_daily_vol) #est_vol for first 2 mins and last 2 mins are not smoothed

    return coef['est_vol']
