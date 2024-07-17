# %%
import pandas as pd

data = pd.read_csv('../datasets/Mod_03_topic_06_diabets_data.csv')

data.head()
data.info()

# %%

X, y = data.drop('Outcome', axis=1), data['Outcome']

# %%
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.scatterplot(x=X['Glucose'], y=X["BMI"], hue=y)

ax.vlines(x=[120,160,], 
          ymin=0, 
          ymax=X['BMI'].max(), 
          color='black', 
          linewidth=0.75)

plt.show()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# %%
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score

clf = tree.DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred)

print(f'Acc.: {acc:.1%}')

# %%
plt.figure(figsize=(80, 15), dpi=196)

tree.plot_tree(clf,
               feature_names=X.columns,
               filled=True,
               fontsize=6,
               class_names=list(map(str, y_train.unique())),
               proportion=True,
               precision=2,
               rounded=True
               )

plt.savefig('../derived/n03_02_decision_tree.png')
plt.show()

# %%
y_train.value_counts(normalize=True)

# %%
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=10)

X_res, y_res = sm.fit_resample(X_train, y_train)

y_res.value_counts(normalize=True)

# %%
clf_upd = tree.DecisionTreeClassifier(random_state=42, 
                                      max_depth=4).fit(X_res, y_res)

y_pred_upd = clf_upd.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred_upd)
print(f'Acc.: {acc:.1%}')

# %%

plt.figure(figsize=(25, 7))

tree.plot_tree(clf_upd,
               feature_names=X.columns,
               filled=True,
               fontsize=8,
               class_names=list(map(str, y_res.unique())),
               proportion=True,
               precision=2,
               rounded=True)
plt.show()

# %%
(pd.Series(
    data=clf_upd.feature_importances_,
    index=X.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()
