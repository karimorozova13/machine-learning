# %%
from sklearn.datasets import load_breast_cancer

data, target  = load_breast_cancer(return_X_y=True, as_frame=True)

data.head()

# %%
#check for nan and dtypes

data.info()

# %%
#check the balance of target
#rule 1:10

target.value_counts()

# %%
from scipy.stats import zscore
import numpy as np

out = data.apply(lambda x: np.abs(zscore(x)).ge(3)).astype(int).mean(1)

out_ind = np.where(out > 0.2)[0]

data.drop(out_ind, inplace=True)
target.drop(out_ind, inplace=True)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=42)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.decomposition import PCA

pca = PCA().set_output(transform='pandas').fit(X_train)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

explained_variance = np.cumsum(pca.explained_variance_ratio_)

ax = sns.lineplot(explained_variance)

ax.set(xlabel='number of components', ylabel='comulative explained variance')

n_components = np.searchsorted(explained_variance, 0.85)

ax.axvline(x=n_components,
           c='teal',
           linestyle='--',
           linewidth=0.75)

plt.show()

# %%
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# %%
X_train_pca.iloc[:, :n_components].head()

# %%
plt.figure(figsize=(8, 8))

ax = plt.subplot(projection='3d')

ax.scatter3D(
    X_train_pca.iloc[:, 0],
    X_train_pca.iloc[:, 1],
    X_train_pca.iloc[:, 2],
    c=y_train,
    s=20,
    cmap='autumn',
    ec='teal',
    lw=0.75)

ax.view_init(elev=30, azim=30)

plt.show()

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)

print(score)

# %%

clf_pca = GradientBoostingClassifier()

clf_pca.fit(X_train_pca.iloc[:, :n_components], y_train)

y_pred_pca = clf_pca.predict(X_test_pca.iloc[:, :n_components])

score_pca = accuracy_score(y_test, y_pred_pca)

print(score_pca)

# %%
import pandas as pd

plt.figure(figsize=(3, 8))

pd.Series(
    data=clf.feature_importances_,
    index=X_train.columns).sort_values(ascending=True).plot.barh()

plt.show()

