# %%
import pandas as pd

data = pd.read_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')

# %%
X, y = (data.drop(['Item_Identifier','Item_Outlet_Sales'], axis=1),
        data['Item_Outlet_Sales'])

# %%
import joblib

with open('../models/mod_07_topic_13_mlpipe.joblib', 'rb') as fl:
    pipe_base = joblib.load(fl)  

# %%
from sklearn.model_selection import cross_val_score
import numpy as np

cv_results = cross_val_score(
    estimator=pipe_base,
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)

rmse_cv = np.abs(cv_results).mean()  

# %%
model_base = pipe_base.fit(X, y)