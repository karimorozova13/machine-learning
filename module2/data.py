import seaborn as sns
import numpy as np
import pandas as pd

planets = sns.load_dataset('planets')
print(planets.shape)

print(planets.head())

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))

print(ser)
print(ser.sum())
print(ser.mean())

df = pd.DataFrame({'A': rng.rand(5),'B': rng.rand(5)})
print(df)
print(df.mean())

print(df.mean(axis='columns'))

print(planets.dropna().describe())

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data': range(6)}, columns=['key', 'data'])
print(df1)
print(df1.groupby('key').sum())
print(planets.groupby('method')['orbital_period'].median())
print(planets.groupby('method')['year'].describe())

rng = np.random.RandomState(0)
df2 = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data1': range(6), 'data2': rng.randint(0, 10, 6)}, columns = ['key', 'data1', 'data2'])

print(df2)
print(df2.groupby('key').agg(['min', 'max', 'median']))

print(df2.groupby('key').agg({'data1': 'min', 'data2': 'max'}))

decade = 10 * (planets['year'] // 10)
decade =  decade.astype(str) + 's'
decade.name = 'decade'

print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))