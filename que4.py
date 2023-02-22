import pandas as pd
import numpy as np
import random

random.seed(0)

col = np.linspace(2, 21, 20, dtype=int)
df = pd.read_csv('21皮尔逊SCORE.csv', sep=',', usecols=col)  # 读取你需要计算的文件
array = df.values
print(df)

# 建立随机数矩阵建立
rand = np.array(np.random.randint(low=0, high=100, size=17) / 100)
rand = rand.reshape(1, len(rand))
rand = np.insert(rand, len(rand), np.random.randint(low=0, high=100, size=17) / 100, axis=0)
rand = np.insert(rand, len(rand), np.random.randint(low=0, high=100, size=17) / 100, axis=0)

e_rand = np.exp(np.sum(rand, axis=0)/3)
e_rand = e_rand.reshape(1, len(e_rand))
col_array = array.shape[1]
e_rand = np.repeat(e_rand,col_array,axis=0).T

array_bar = array * e_rand

# 保存
sf = pd.read_csv('./source/21皮尔逊48.csv', sep=',')
sf_add = pd.DataFrame(data=array_bar, columns=df.columns)
sf = pd.concat([sf_add], axis=1)
sf.to_csv('国家打分修正.csv')

