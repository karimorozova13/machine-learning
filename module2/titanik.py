import seaborn as sns
import pandas as pd

titanicD = sns.load_dataset('titanic')
print(titanicD.head())

print(titanicD.groupby(['sex', 'class'], observed=True)[['survived']].mean().unstack())

print(titanicD.pivot_table('survived', index='sex', columns='class'))

age = pd.cut(titanicD['age'], [0, 18, 80])
print(titanicD.pivot_table('survived', ['sex', age], 'class'))

fare = pd.qcut(titanicD['fare'], 2)
print(titanicD.pivot_table('survived', ['sex', age], [fare, 'class'], observed=True))