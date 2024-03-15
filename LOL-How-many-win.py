# Load packages and dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline 해당 코드는 jupyter에서 작동되는 코드
sns.set_style('darkgrid')

df = pd.read_csv('high_diamond_ranked_10min.csv')
df.head()

# EDA
# check missing values and data type
df.info()  # int or float

df_clean = df.copy()

# Drop some unecessary columns. e.g. blueFirstblood/redfirst blood blueEliteMonster/redEliteMonster blueDeath/redKills etc are repeated
# 불필요한 변수 삭제
# Based on personal experience with the game, mimion yield gold+experience, we can drop minion kill too
# 미니언 수율 골드+경험치를 바탕으로 미니언 킬도 드롭할 수 있습니다.

cols = ['gameId', 'redFirstBlood', 'redKills', 'redEliteMonsters', 'redDragons', 'redTotalMinionsKilled',
        'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin', 'redHeralds',
        'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin', 'blueTotalMinionsKilled']
df_clean = df_clean.drop(cols, axis=1)

df_clean.info()

# Next let's check the relationship between parameters of blue team features
# 블루팀의 매개변수를 확인
g = sns.PairGrid(data=df_clean, vars=["blueKills", "blueAssists", "blueWardsPlaced", "blueTotalGold"], hue='blueWins',
                 palette='Set1')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
plt.show()

# 많은 공선성을 보인다.
# 공선성(collinearity): 하나의 독립변수가 다른 하나의 독립변수로 잘 예측되는 경우, 또는 서로 상관이 높은 경우

# We can see that a lot of the features are highly correlated, let's get the correlation matrix
plt.figure(figsize=(16, 12))
sns.heatmap(df_clean.drop('blueWins', axis=1).corr(), cmap='YlGnBu', annot=True, fmt='.2f', vmin=0)
plt.show()

# Based on the correlation matrix, let's clean the dataset a little bit more to avoid colinearity
cols = ['blueAvgLevel', 'redWardsPlaced', 'redWardsDestroyed', 'redDeaths', 'redAssists', 'redTowersDestroyed',
        'redTotalExperience', 'redTotalGold', 'redAvgLevel']
df_clean = df_clean.drop(cols, axis=1)

# Next let's drop the columns has little correlation with bluewins
corr_list = df_clean[df_clean.columns[1:]].apply(lambda x: x.corr(df_clean['blueWins']))
cols = []
for col in corr_list.index:
    if (corr_list[col] > 0.2 or corr_list[col] < -0.2):
        cols.append(col)
cols
