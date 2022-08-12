import pandas as pd
import numpy as np
import baseline.main_model as baseline
import statsmodels.formula.api as smf
from datetime import datetime
from dateutil.relativedelta import relativedelta

def PriceVolatilityModel(daily_data, intraday_data, px_df, real_df, test_date, train_date):
    '''Intraday volume model'''

    train_df = calculate_residuals(intraday_data, daily_data, real_df, px_df, train_date)

    reg = smf.ols(formula="residual ~ pr_sd", data=train_df).fit()

    test_df = calculate_residuals(intraday_data, daily_data, real_df, px_df, test_date)
    test_df['vol_est'] = round(test_df['baseline'] + reg.params[1] * test_df['pr_sd'].values + reg.params[0])
    test_df['error'] = np.abs(test_df['vol_est'] - test_df['real']) / test_df['real']
    test_df['baseline_error'] = np.abs(test_df['baseline'] - test_df['real']) / test_df['real']
    test_df['error_improvement'] = test_df['baseline_error'] - test_df['error']

    test_df.fillna(0, inplace=True)

    return test_df

def compare_stocks():
    '''Evaluates all stocks in Russell 3000 Sample dataset'''
    with open("data/russell/Russell_Sample.txt", 'r') as input:
        tickers = input.read().splitlines()

    error_comparison = None

    for ticker in tickers:

        try:
            error_comparison = evaluation(ticker, error_comparison)
        except:
            pass

    error_comparison.to_csv('data/volatility_results/russell_error_improvement.csv')

def evaluation(ticker, error_comparison):
    '''Output prediction and error results for PriceVolatilityModel on a ticker'''
    daily_data, intraday_data, px_df, real_df, test_dates = preprocess(ticker)

    prediction_df = pd.DataFrame(index=px_df.index.values[1:])
    error_df = pd.DataFrame(index=px_df.index.values[1:])
    baseline_error_df = pd.DataFrame(index=px_df.index.values[1:])
    error_improvement_df = pd.DataFrame(index=px_df.index.values[1:])

    avg_improvement = []

    for i in range(1, len(test_dates)):

        test_date = test_dates[i]
        train_date = test_dates[i-1]

        results = PriceVolatilityModel(daily_data, intraday_data, px_df, real_df, test_date, train_date)

        prediction_df[test_date] = results['vol_est']
        error_df[test_date] = results['error']
        baseline_error_df[test_date] = results['baseline_error']
        error_improvement_df[test_date] = results['error_improvement']

        avg_improvement.append(results['error_improvement'].mean())

    prediction_df.to_csv(f"data/volatility_results/predictions/predictions_{ticker}.csv")
    error_df.to_csv(f"data/volatility_results/errors/errors_{ticker}.csv")
    baseline_error_df.to_csv(f"data/volatility_results/baseline_errors/baseline_errors_{ticker}.csv")
    error_improvement_df.to_csv(f"data/volatility_results/error_improvement/error_improvement_{ticker}.csv")

    if error_comparison is None:
        error_comparison = pd.DataFrame(index=test_dates[1:])

    error_comparison[ticker] = avg_improvement

    print(f"{ticker}: {sum(avg_improvement)/len(avg_improvement)}")

    return error_comparison

def preprocess(ticker):
    '''Preprocess data for model evaluation'''
    daily_file = f"data/russell/russell_daily.csv"
    intraday_file = f"data/russell/russell_{ticker}.csv"
    price_file = f"data/prices/Px_{ticker}.csv"

    daily_data, intraday_data = preprocess_intraday_daily(intraday_file, daily_file, ticker)

    px_data = preprocess_px(price_file)
    px_df = px_data.pivot(index='time', columns='date', values='pr_sd')
    px_df.fillna(0, inplace=True)
    px_df.columns = px_data['date'].dt.date.unique()

    real_vol = intraday_data.copy()
    real_vol.index = pd.DatetimeIndex(real_vol['datetime'])
    real_vol['time'] = real_vol.index.time

    real_vol = real_vol.between_time('09:45:00', '15:59:00')
    real_df = real_vol.pivot(index='time', columns='date', values='size')
    real_df.fillna(0, inplace=True)
    real_df.columns = intraday_data['date'].dt.date.unique()

    test_dates = sorted(px_data['date'].dt.date.unique())

    return daily_data, intraday_data, px_df, real_df, test_dates

def preprocess_px(price_file):
    '''Preprocess price data'''
    px_data = pd.read_csv(price_file)

    px_data = baseline.format_intraday(px_data)
    px_data = baseline.eliminate_half_days(px_data)

    px_data.index = pd.DatetimeIndex(px_data['datetime'])

    px_data['time'] = px_data.index.time

    px_data.drop(columns=['datetime', 'sym_root', 'sym_suffix'], inplace=True)

    px_data = px_data.between_time('09:30:00', '16:00:00')

    px_data['pct_prx'] = px_data['price'].pct_change().round(decimals = 4) + 1
    px_data['pr_sd'] = px_data['pct_prx'].rolling('15T').std()

    px_data = px_data.between_time('09:45:00', '15:59:00')

    return px_data

def preprocess_intraday_daily(intraday_file, daily_file, ticker):
    '''Preprocess intraday and daily volume data'''
    daily_data = pd.read_csv(daily_file, parse_dates=['DATE'])
    intraday_data = pd.read_csv(intraday_file)

    intraday_data = baseline.format_intraday(intraday_data)

    daily_data = daily_data.loc[daily_data['symbol'] == ticker]
    intraday_data = intraday_data.loc[intraday_data['symbol'] == ticker]

    return daily_data, intraday_data

def calculate_residuals(intraday_data, daily_data, real_df, px_df, date):
    '''Calls on baseline model and returns residual dataframe'''
    base = calculate_baseline(intraday_data, daily_data, date)
    df = base.to_frame(name='baseline')
    df['real'] = real_df[date]
    df['residual'] = df['real'] - df['baseline']

    df = df[14:]  # excludes times before 9:45AM
    df['pr_sd'] = px_df[date]
    
    return df
    
def calculate_baseline(intraday_data, daily_data, date):
    '''Get baseline model results'''
    overnight_gap = baseline.get_overnight_gap(daily_data, date)

    daily_data = daily_data.loc[
        (daily_data['DATE'].dt.date >= (date - relativedelta(years=2))) & (daily_data['DATE'].dt.date < date)]
    intraday_data = intraday_data.loc[
        (intraday_data['date'].dt.date >= (date - relativedelta(years=2))) & (intraday_data['date'].dt.date < date)]

    estimated_daily = baseline.DailyModel(daily_data)
    estimated_intraday = baseline.IntradayModel(intraday_data, daily_data, estimated_daily, overnight_gap)

    return estimated_intraday

if __name__ == '__main__':
    compare_stocks()