import pandas as pd
import numpy as np


col = np.linspace(1, 16, 16, dtype=int)
# 读取数据
df = pd.read_csv(r'./source/剔除G9后数据.csv', sep=',', header='infer', usecols=col)

corr = df.corr(method='pearson')
corr.to_csv("剔除G9后数据皮尔逊.csv")
print(corr)

