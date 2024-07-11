import pandas as pd

df = pd.read_csv('../datasets/mod_02_topic_04_ts_data.csv')
print(df.head())

# %%
df['ds'] = pd.to_datetime(df['ds'])
df = df.set_index('ds').squeeze()

print(df.describe())

# %%
import numpy as np

df = np.log(df)

print(df.head())

# %%
df_hist = df.iloc[:-365]
df_test = df.iloc[-365:]

print(df_hist.isna().sum())

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

fig, ax = plt.subplots(figsize=(30, 7))

ax.vlines(
    x=df_hist.index,
    ymin=0,
    ymax=df_hist,
    linewidth=0.5,
    color='blue'
    )

# %%

df_hist = df_hist.asfreq('D').interpolate()

print(df_hist.isna(). sum())

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(np.arange(len(df_hist)).reshape(-1,1), df_hist)

trend = model.predict(np.arange(len(df_hist)).reshape(-1,1))

ax = plt.subplots(figsize=(10,3))

sns.scatterplot(df_hist)
sns.lineplot(y=trend, x=df_hist.index, c='tomato')

# %%
df_mod = df_hist - trend + trend.mean()

sns.catplot(
            y=df_hist,
            x=df_hist.index.month,
            kind='box',
            showfliers=False)

# %%
import statsmodels.api as sm

decomp = sm.tsa.seasonal_decompose(df_hist)
decomp_plot = decomp.plot()

# %%
from scipy.stats import zscore

df_zscore = zscore(decomp.resid, nan_policy='omit')

# %%
def zscore_adv(x, window):
    r=x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s 
    return z

df_zscore_adv = zscore_adv(decomp.resid, window=7)

# %%

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(10,7))

for i, d in enumerate([df_zscore, df_zscore_adv]):
    ax =axes[i]
    sns.lineplot(d, ax=ax)
    ax.fill_between(d.index.values, -3, 3, alpha=0.15)
    
# %%

playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2013-01-12',
                          '2014-01-12',
                          '2014-01-19',
                          '2014-02-02',
                          '2015-01-11',
                          '2016-01-17']),
    'lower_window':0,
    'upper_window':1
    })

superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2014-02-02']),
    'lower_window': 0,
    'upper_window': 1})

holidays = pd.concat((playoffs, superbowls)).reset_index(drop=True)

print(holidays)

# %%

outliers = np.where(~df_zscore_adv.between(-3,3) * df_zscore_adv.notna())[0]

outliers = list(set(df_hist.index[outliers]).difference(holidays['ds']))

fig, ax = plt.subplots(figsize=(10, 3))
sns.lineplot(df_hist, ax=ax)
sns.scatterplot(
    x=outliers,
    y=df_hist[outliers],
    color='red',
    ax=ax
    ) 

# %%

df_hist.loc[outliers] = np.nan

df_hist = df_hist.interpolate()

# %%
from prophet import Prophet

df_hist = df_hist.reset_index()

mp = Prophet(holidays=holidays)
mp.add_seasonality(name='yearly', period=365, fourier_order=3)

mp.fit(df_hist)

# %%

future = mp.make_future_dataframe(freq='D', periods=365)

forecast = mp.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# %%
mp.plot_components(forecast)
mp.plot(forecast)

# %%
pred = forecast.iloc[-365:][['ds', 'yhat']]

fig, ax = plt.subplots(figsize=(20, 5))

ax.vlines(
    x=df_test.index,
    ymin=5,
    ymax=df_test,
    linewidth=0.75,
    label='fact',
    zorder=1
    )

ax.vlines(
    x=df_test[df_test.index.isin(holidays['ds'])].index,
    ymin=5,
    ymax=df_test[df_test.index.isin(holidays['ds'])],
    linewidth=0.75,
    label='special events',
    color='tomato',
    zorder=2
    )

sns.lineplot(data=pred, y='yhat', x='ds', c='green', label='prophet', ax=ax)

ax.margins(x=0.01)

plt.show()

# %%
from sklearn.metrics import median_absolute_error

approx_mape = median_absolute_error(df_test, pred['yhat'])
print(f'Accuracy: {1 - approx_mape:.1%}')