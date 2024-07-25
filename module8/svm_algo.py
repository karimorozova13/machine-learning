import pandas as pd

data = pd.read_csv('../datasets/mod_04_topic_08_petfinder_data.csv.gz')

# %%
data.info()

# %%

data.nunique()

# %%
data['Description'].head()
data.drop('Description', axis=1, inplace=True)

# %%
data['AdoptionSpeed'].value_counts().sort_index()

# %%
import numpy as np

data['AdoptionSpeed'] = np.where(data['AdoptionSpeed'] == 4, 0, 1)
data['AdoptionSpeed'].value_counts()

# %%
data['Fee'] = data['Fee'].astype(bool).astype(int).astype(str)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('AdoptionSpeed', axis=1),
                                                    data['AdoptionSpeed'],
                                                    test_size=0.2,
                                                    random_state=42)

# %%
num_columns = X_train.select_dtypes(exclude='object').columns


# %%
# tranform numeric columns to discret

from sklearn.preprocessing import KBinsDiscretizer

kbins = KBinsDiscretizer(encode='ordinal').fit(X_train[num_columns])

X_train[num_columns] = kbins.transform(X_train[num_columns]).astype(int).astype(str)

X_test[num_columns] = kbins.transform(X_test[num_columns]).astype(int). astype(str)

# %%
X_train.select_dtypes(include='object')

# %%
#encode categorial values

import category_encoders as ce

encoder = ce.TargetEncoder()

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

X_train.head()

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.svm import SVC

clf = SVC(class_weight='balanced',
          kernel='poly',
          probability=True,
          random_state=42)
    
clf.fit(X_train, y_train)

# %%
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = clf.predict(X_test)

mtx = confusion_matrix(y_test, y_pred)

# %%
print(f'Model accuracy is: {accuracy_score(y_test, y_pred):.1%}')

# %%
pet = pd.DataFrame(data={
    'Type': 'Dog',
    'Age': 1,
    'Breed1': 'Kari',
    'Gender': 'Female',
    'Color1': 'Black',
    'Color2': 'White',
    'MaturitySize': 'Small',
    'FurLength': 'Short',
    'Vaccinated': 'No',
    'Sterilized': 'No',
    'Health': 'Serious Injury',
    'Fee': True,
    'PhotoAmt': 4,
    }, index=[0])

# %%
pet[num_columns] = kbins.transform(pet[num_columns]).astype(int).astype(str)

prob = clf.predict_proba(scaler.transform(encoder.transform(pet))).flatten()
                         
print(f'This pet has a {prob[1]:.1%} probability "of getting adopted"')
