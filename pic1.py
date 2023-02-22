# 导入库函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

col = np.linspace(1, 2, 2, dtype=int)
# 读取数据
df = pd.read_csv(r'./source/拟合.csv', sep=',', usecols=col)

print(df)

x = np.linspace(10, 100, 2000, dtype=float)
delta = np.linspace(2, 6, 2000, dtype=float)
y= 28.802* np.power(x,0.2454)
y1 = y + delta            # 生成第二条曲线
y2 = y - delta            # 生成第二条曲线


ax1=plt.gca()
ax1.patch.set_facecolor("#EAEBF2")    # 设置 ax1 区域背景颜色
ax1.patch.set_alpha(1)
plt.grid(c='w')
plt.plot(x, y, '#6FAE7D')
plt.plot(df['Goal 9 Score'], df['Goal 3 Score'], 'o', color='#6FAE7D')
plt.fill_between(x, y1, y2, facecolor='#D5E1DF', alpha=0.9)
plt.rc('font',family='Times New Roman')
plt.xlabel("SDG9")
plt.ylabel("SDG6")
plt.show()
