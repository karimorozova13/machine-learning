# %%
import pandas as pd
import os

# Change the working directory to the location of rain.py
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Verify the current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Define the relative path to the file from the location of rain.py
file_path = './mod_03_topic_05_weather_data.csv.gz'

# Ensure the file exists at the specified path
if os.path.exists(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    print("File read successfully.")
    # Optionally, display the first few rows of the dataframe
    print(df.head())
else:
    print("File not found:", file_path)

# %%
import pandas as pd

data = pd.read_csv('../datasets/mod_04_topic_07_bank_data.csv', sep=';')

# %%
data.shape


# %%
data.drop('duration', axis=1, inplace=True)

# %%
data.describe()

# %%
d = data.skew(numeric_only=True)
print(d)

# %% 
from scipy.stats import zscore

data = data[zscore(data['campaign']).abs().lt(2)]

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

cor_mtx = data.drop('y', axis=1).corr(numeric_only=True).abs()

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(cor_mtx,
            cmap='crest',
            annot=True,
            fmt=".2f",
            linewidth=.5,
            mask=np.triu(np.ones_like(cor_mtx, dtype=bool)),
            cbar=False,
            ax=ax)

plt.show()

# %%
data.drop([
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'nr.employed'
    ], axis=1, inplace=True)

# %%
data.select_dtypes(include='object').nunique()

# %%
data['y'] = data['y'].replace({'no': 0, 'yes': 1})

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('y', axis=1),
                                                    data['y'],
                                                    random_state=42,
                                                    test_size=0.2
                                                    )

# %%
#replace categorial values with numeric

import category_encoders as ce

cat_col = X_train.select_dtypes(include='object').columns
cat_col

encoder = ce.WOEEncoder(cols=cat_col)

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %% 
# normilize data, decrease asymetry

from sklearn.preprocessing import PowerTransformer

power_transform = PowerTransformer().set_output(transform='pandas')

X_train = power_transform.fit_transform(X_train)

X_test = power_transform.transform(X_test)

# %%
X_train.skew()

# %%
y_train.value_counts(normalize=True)

# %%
#balanced the target 50/50

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=50)

X_res, y_res = sm.fit_resample(X_train, y_train)

# %%
#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

knn_classifier = KNeighborsClassifier(n_neighbors=15, n_jobs=1).fit(X_res, y_res)

y_pred = knn_classifier.predict(X_test)
knn_score = balanced_accuracy_score(y_test, y_pred)

print(f'KNN model accuracy: {knn_score:.1%}')

# %%
from sklearn.naive_bayes import GaussianNB

bayes_mod = GaussianNB().fit(X_res, y_res)

y_bayes_predict = bayes_mod.predict(X_test)

gnb_score  = balanced_accuracy_score(y_test, y_bayes_predict)

print(f'GNB  model accuracy: {gnb_score :.1%}')


# %%
from sklearn.metrics import confusion_matrix

knn_mtx = confusion_matrix(y_test, y_pred)
gnb_mtx =confusion_matrix(y_test, y_bayes_predict)
print(knn_mtx)
print(gnb_mtx)
