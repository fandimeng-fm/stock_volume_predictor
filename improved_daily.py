import pandas as pd
import numpy as np
import baseline.main_model as baseline
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import stats
from dateutil.relativedelta import relativedelta
import warnings
import csv

warnings.filterwarnings('ignore')

daily_file = 'data/russell/russell_daily.csv'
vix_file = 'data/vix_prices.csv'

def CombinedVIXModel(data, overnight_gap, vix, pred_log_vol):
    '''Daily model incorporating robust ARMA and residual regression on overnight price gap and VIX'''
    gap_pct = stats.percentileofscore(data['overnight_gap'], overnight_gap) / 100

    if gap_pct < 0.9:
        overnight_gap = None

    pred_log_vol += ols_VIX(data, overnight_gap, vix)
    prediction = round(np.exp(pred_log_vol))

    return prediction

def preprocess(daily_file, vix_file, ticker):
    '''Preprocess data'''
    daily_data = pd.read_csv(daily_file, parse_dates=['DATE'])

    daily_data = daily_data.loc[daily_data['symbol'] == ticker]
    daily_data['log_vol'] = np.log(daily_data['total_vol_m'])

    daily_data = calc_overnight_gap(daily_data)

    daily_data = add_vix_data(daily_data, vix_file)

    daily_data.set_index('DATE', drop=False, inplace=True)

    return daily_data


def calc_overnight_gap(daily_data):
    '''Calculate the overnight price gap for prediction date.'''

    daily_data['price_diff'] = abs(daily_data['OPrc'] - daily_data['CPrc'].shift())
    daily_data['overnight_gap'] = daily_data['price_diff'] / daily_data['CPrc'].shift()
    daily_data['gap_pct'] = daily_data['overnight_gap'].rank(pct=True)

    return daily_data


def add_vix_data(daily_data, vix_file):
    '''Merge VIX data with daily volume data'''
    vix = pd.read_csv(vix_file)
    vix['Date'] = pd.to_datetime(vix['Date'], format='%Y%m%d')
    vix.fillna(method='ffill', inplace=True)
    vix['vix'] = vix[
        'vix'].shift()  # since vix values are close of day price, shift so date associated with vix of day prior
    daily_data = daily_data.merge(vix, left_on='DATE', right_on='Date', how='left')
    daily_data.drop(columns=['Date'], inplace=True)

    return daily_data


def evaluation(daily_file, vix_file, ticker):
    '''Evaluate model performance for 2021 (1 year period) on a given stock.'''
    daily_data = preprocess(daily_file, vix_file, ticker)

    eval_df = daily_data.loc[daily_data.index.year == 2021]
    eval_df = eval_df[['total_vol_m']]

    base = []
    combinedVIX = []

    for date in eval_df.index:
        training = daily_data.loc[
            (daily_data.index.date >= (date - relativedelta(years=2))) & (daily_data.index.date < date)].copy()

        training['smooth_log_vol'] = smooth(training['log_vol'])

        pred_smooth, fitted_smooth = ARMA(training['smooth_log_vol'])

        pred_normal, fitted_normal = ARMA(training['log_vol'])

        training['pred_log_vol'] = fitted_normal

        overnight_gap = daily_data.loc[date, 'overnight_gap']
        vix = daily_data.loc[date, 'vix']

        base.append(baseline.Model1(training))
        combinedVIX.append(CombinedVIXModel(training, overnight_gap, vix, pred_smooth))

    eval_df['baseline'] = base
    eval_df['combinedVIX_model'] = combinedVIX

    eval_df['baseline_error'] = abs(eval_df['total_vol_m'] - eval_df['baseline']) / eval_df['total_vol_m']
    eval_df['combinedVIX_error'] = abs(eval_df['total_vol_m'] - eval_df['combinedVIX_model']) / eval_df['total_vol_m']

    return eval_df


def compare_stocks():
    '''Evaluate performance on Russell 3000 sample data set.'''
    output_headers = ['ticker', 'baseline', 'robust_overnight_vix']

    with open("data/russell/Russell_Sample.txt", 'r') as input:
        tickers = input.read().splitlines()

    output = open("data/daily_compare/model_comparison.csv", 'w')
    writer = csv.DictWriter(output, fieldnames=output_headers)
    writer.writeheader()
    output.close()

    for ticker in tickers:
        try:
            df = evaluation(daily_file, vix_file, ticker)

            ticker_results = {}
            ticker_results['ticker'] = ticker
            ticker_results['baseline'] = df['baseline_error'].mean()
            ticker_results['robust_overnight_vix'] = df['combinedVIX_error'].mean()

            output = open("data/daily_compare/model_comparison.csv", 'a')
            writer = csv.DictWriter(output, fieldnames=output_headers)
            writer.writerow(ticker_results)
            output.close()

            df.to_csv(f"data/daily_compare/daily_compare_{ticker}.csv")
        except:
            pass


def smooth(series):
    '''Implements robust ARMA regularization on list of a series of volume data.'''
    res = series.copy()
    n = series.shape[0]
    Forward = series.diff().abs().describe()
    Backward = series.diff(-1).abs().describe()
    F = 1.5 * (Forward['75%'] - Forward['25%'])
    B = 1.5 * (Backward['75%'] - Backward['25%'])

    for i in range(1, n):
        if i != n - 1:
            if series.iloc[i] - series.iloc[i - 1] >= F and series.iloc[i] - series.iloc[i + 1] >= B:
                res.iloc[i] = (series.iloc[i - 1] + series.iloc[i + 1]) / 2
        else:
            if series.iloc[i] - series.iloc[i - 1] >= F:
                res.iloc[i] = res.iloc[i - 1] + F

    return res


def ARMA(log_vol):
    '''Implements ARMA (1,1) model and returns prediction for next time period and fitted values'''
    model = ARIMA(log_vol, order=(1, 0, 1))
    model_fit = model.fit()
    pred_log_vol = model_fit.forecast(1).iloc[0]
    fitted = model_fit.fittedvalues

    return pred_log_vol, fitted

def ols_VIX(data, overnight_gap, vix):
    '''Implements residual regression on overnight price gap and prior closing VIX price.'''
    residuals = data[['log_vol', 'vix', 'overnight_gap', 'gap_pct', 'pred_log_vol']]

    residuals['res'] = residuals['log_vol'] - residuals['pred_log_vol']

    if overnight_gap is not None:
        ols_data = residuals.loc[residuals['gap_pct'] > 0.9]
        X = sm.add_constant(ols_data[['overnight_gap', 'vix']])
    else:
        ols_data = residuals
        X = sm.add_constant(ols_data['vix'])

    y = ols_data['res']
    ols_model = sm.OLS(y, X)
    ols_fit = ols_model.fit()

    if overnight_gap is not None:
        pred_adj = ols_fit.params[0] + ols_fit.params[1] * overnight_gap + ols_fit.params[2] * vix
    else:
        pred_adj = ols_fit.params[0] + ols_fit.params[1] * vix

    return pred_adj


if __name__ == '__main__':
    compare_stocks()