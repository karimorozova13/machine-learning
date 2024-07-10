from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

data = california_housing['frame']

# %%
print(data.head())

# %%
target = data.pop('MedHouseVal')
print(target.head())

# %%
print(data.info())

# %%
import seaborn as sns
import pandas as pd

sns.set_theme()
melted = pd.concat([data, target], axis=1). melt()

g = sns.FacetGrid(melted, col='variable', col_wrap=3, sharex=False, sharey=False)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

# %%
features_of_interest = ['AveRooms', "AveBedrms", 'AveOccup', 'Population']

print(data[features_of_interest].describe())

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))

sns.scatterplot(data=data, 
                x='Longitude', 
                y='Latitude', 
                size=target, 
                hue=target, 
                palette='viridis',
                alpha=0.5,
                ax=ax
                )

plt.legend(title='MedHouseVal',bbox_to_anchor=(1.05, 0.95),loc='upper left')

plt.title('Median house value depending of\n their spatial location')

plt.show()

# %%
import numpy as np

columns_drop = ['Longitude', 'Latitude']
subset = pd.concat([data, target], axis=1).drop(columns=columns_drop)

corr_mtx = subset.corr()

mask_mtx = np.zeros_like(corr_mtx)

np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7,6))

sns.heatmap(subset.corr(), 
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
print(X_train_scaled.describe())

# %%
from sklearn.linear_model import LinearRegression

model  =LinearRegression().fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

ymin, ymax = y_train.agg(['min', 'max']).values

y_pred = pd.Series(y_pred, index=X_test_scaled.index).clip(ymin, ymax)

print(y_pred.head())

# %%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# %%
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures().set_output(transform='pandas')

Xtr = poly.fit_transform(X_train_scaled)
Xts = poly.fit_transform(X_test_scaled)

model_upd = LinearRegression().fit(Xtr, y_train)
y_pred_upd = model_upd.predict(Xts)
y_pred_upd = pd.Series(y_pred_upd, index=Xts.index).clip(ymin, ymax)

r_sq_upd = model_upd.score(Xtr, y_train)
mae_upd = mean_absolute_error(y_test, y_pred_upd)
mape_upd = mean_absolute_percentage_error(y_test, y_pred_upd)

print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')