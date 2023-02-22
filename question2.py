import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

col = np.linspace(1, 21, 21, dtype=int)
df = pd.read_csv('./source/国家修正熵权输入.csv', sep=',', usecols=col)  # 读取你需要计算的文件
array = df.values
print(df)

# 归一化
tool = MinMaxScaler(feature_range=(0, 1))
array_bar = tool.fit_transform(array)
array_bar = np.transpose(array_bar)
print(array_bar)

# 计算p
p = []
for i in range(0, len(array_bar)):
    p.append((array_bar[i] / sum(array_bar[i]) + 0.00000000001).tolist())
p = np.array(p)

# 计算熵值
k = 1 / math.log(len(p))
e = -k * np.sum(p * np.log(p), axis=1)

# 计算信息熵冗余度
d = e * -1 + 1
w = d / np.sum(d)

# 计算得分
array_bar = np.transpose(array_bar)
s = np.sum(100 * w * array_bar, axis=1)

# 保存
sf = pd.read_csv('./source/国家修正熵权输入.csv', sep=',')
sf_add = pd.DataFrame(data=s, columns=['SCORE'])
sf = pd.concat([sf, sf_add], axis=1)
sf.to_csv('Q4国家修正熵权SCORE.csv')
