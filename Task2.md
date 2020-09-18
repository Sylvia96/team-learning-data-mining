# EDA

## 1. 学习目标

* 学习如何对数据集整体概况进行分析，包括数据集的基本情况（缺失值，异常值），并对数据集进行验证是否可以进行接下来的机器学习或者深度学习建模.

* 了解变量间的相互关系、变量与预测值之间的存在关系。

* 为特征工程做准备。

## 2. 学习内容提纲

（1）数据**整体**了解

* 读取数据集 —— pd.read_csv

* 了解数据集大小和原始特征维度 —— shape

* 熟悉数据类型 —— info

* 粗略查看数据集中各特征基本统计量

(2) **缺失值/唯一值**重点了解

* 查看数据缺失值情况

* 查看唯一值特征情况

（3）数据**类型**（深入数据）

* 类别型数据

* 数值型数据

  * 离散数值型数据
  
  * 连续数值型数据

（4）数据间相关**关系**

* 特征和特征之间关系

* 特征和目标变量之间关系

(5) 生成数据**报告** —— pandas_profiling

## 3. coding notes
### 3.0 平台 
Google Colaboratory
### 3.1 读取数据集

```python
# 导入数据分析和可视化库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据集读取
data_train = pd.read_csv('/content/train.csv')
data_test_a = pd.read_csv('/content/testA.csv')

# 部分读取
data_train_sample = pd.read_csv('/content/train.csv', nrows=5)

# 分块读取
chunker = pd.read_csv("./train.csv",chunksize=5)
for item in chunker:
    print(type(item))
    print(len(item))
```
