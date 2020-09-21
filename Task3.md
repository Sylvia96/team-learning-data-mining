# 特征工程

## 1. 内容提纲
- 数据预处理
  - 缺失值的填充
  - 时间格式处理
  - 对象类型特征转换到数值
- 异常值处理
  - 基于3segama原则
  - 基于箱型图
- 数据分箱
  - 固定宽度分箱
  - 分位数分箱
    - 离散数值型数据分箱
    - 连续数值型数据分箱
  - 卡方分箱（选做作业）
- 特征交互
  - 特征和特征之间组合
  - 特征和特征之间衍生
  - 其他特征衍生的尝试（选做作业）
- 特征编码
  - one-hot编码
  - label-encode编码
- 特征选择
    - 1 Filter
    - 2 Wrapper （RFE）
    - 3 Embedded

## 2. 数据导入
```python
# 导入包并读取数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')

data_train =pd.read_csv('../train.csv')
data_test_a = pd.read_csv('../testA.csv')
````

## 3. 数据预处理
```python
# 查找出数据中的对象特征和数值特征
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
```

* 缺失值填充

  * 把所有缺失值替换为指定的值0

  * data_train = data_train.fillna(0)

  * 向用缺失值上面的值替换缺失值

  * data_train = data_train.fillna(axis=0,method='ffill')

  * 纵向用缺失值下面的值替换缺失值,且设置最多只填充两个连续的缺失值

  * data_train = data_train.fillna(axis=0,method='bfill',limit=2)
  
```python
#查看缺失值情况
data_train.isnull().sum()

#按照平均数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
#按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())

data_train.isnull().sum()

#查看类别特征
#对象型类别特征需要进行预处理，其中['issueDate']为时间格式特征。
category_fea
```

