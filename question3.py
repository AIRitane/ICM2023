import pandas as pd
import numpy as np


col = np.linspace(1, 17, 17, dtype=int)
# 读取数据
df = pd.read_csv(r'./国家打分修正.csv', sep=',', header='infer', usecols=col)

corr = df.corr(method='pearson')
corr.to_csv("国家打分修正皮尔逊.csv")
print(corr)

