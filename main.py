import pingouin as pg
import pandas as pd
import numpy as np
import random
import csv

# 提供随机种子
random.seed(0)


# 得到数量为num的某年选取的行数索引
def get_row_select(num, years):
    row = np.linspace(2 + years, 3874, 177, dtype=int)
    row = row.tolist()
    if num >= len(row):
        return None

    index = random.sample(row, num)
    return index


def main(i):
    # 产生选择的行列
    col = np.linspace(7, 23, 17, dtype=int)
    row = get_row_select(20, 0)

    # 读取数据
    df = pd.read_csv(r'./source/Backdated SDG Index.csv', sep=',', header='infer', usecols=col)
    country = pd.read_csv(r'./source/Backdated SDG Index.csv', sep=',', header='infer', usecols=[1]).values[row]
    country = country.flatten()
    array = df.values[0::, 0::]
    array = array[row]

    df = pd.DataFrame(data=array, columns=df.columns, dtype=float, index=country)

    # 对选择的数据做偏相关
    pcorr = df.pcorr().round(3)
    corr = df.corr(method='pearson')

    print(pcorr)
    print(corr)
    #

    df.to_csv('原始选择数据{}.csv'.format(i))
    pcorr.to_csv('偏相关{}.csv'.format(i))
    corr.to_csv('皮尔逊相关{}.csv'.format(i))


for i in range(0, 200):
    random.seed(i)
    main(i)
