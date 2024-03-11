# Load packages and dataset
import inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline 해당 코드는 jupyter에서 작동되는 코드
sns.set_style('darkgrid')

df = pd.read_csv('high_diamond_ranked_10min.csv')
df.head()