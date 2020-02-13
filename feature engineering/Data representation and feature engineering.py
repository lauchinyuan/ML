import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures  # 多项式特征
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.preprocessing import MinMaxScaler
# 分类特征也叫离散特征
# 对于特定应用来说，如何找到最佳数据表示，这个问题被称为特征工程
# 用正确的方式表示数据，对监督模型性能的影响比所选的参数还要大

# One-Hot编码（虚拟变量、N取一编码）
# 虚拟变量背后的思想是将一个分类变量替换为一个或多个新特征
# 新特征取值分别为0和1
data = pd.read_csv(r'D:\github\ML\data\adult.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])

# 为了便于说明，我们只选择了其中几列
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
print(data.head())
#    age          workclass  ...          occupation  income
# 0   39          State-gov  ...        Adm-clerical   <=50K
# 1   50   Self-emp-not-inc  ...     Exec-managerial   <=50K
# 2   38            Private  ...   Handlers-cleaners   <=50K
# 3   53            Private  ...   Handlers-cleaners   <=50K
# 4   28            Private  ...      Prof-specialty   <=50K
# [5 rows x 7 columns]

print(data.gender.value_counts())
#  Male      21790
#  Female    10771
# Name: gender, dtype: int64

# 数据变换
print('Original features:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Feature after get_dummies:\n', list(data_dummies.columns))
# Original features:
#  ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']
#
# Feature after get_dummies:
#  ['age', 'hours-per-week', 'workclass_ ?',
#  'workclass_ Federal-gov', 'workclass_ Local-gov',
#  'workclass_ Never-worked', 'workclass_ Private',
#  'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc',
#  'workclass_ State-gov', 'workclass_ Without-pay',
#  'education_ 10th', 'education_ 11th', 'education_ 12th',
#  'education_ 1st-4th', 'education_ 5th-6th',
#  'education_ 7th-8th', 'education_ 9th',
#  'education_ Assoc-acdm', 'education_ Assoc-voc',
#  'education_ Bachelors', 'education_ Doctorate',
#  'education_ HS-grad', 'education_ Masters',
#  'education_ Preschool', 'education_ Prof-school',
#  'education_ Some-college', 'gender_ Female',
#  'gender_ Male', 'occupation_ ?',
#  'occupation_ Adm-clerical',
#  'occupation_ Armed-Forces',
#  'occupation_ Craft-repair', 'occupation_ Exec-managerial',
#  'occupation_ Farming-fishing',
#  'occupation_ Handlers-cleaners',
#  'occupation_ Machine-op-inspct',
#  'occupation_ Other-service', 'occupation_ Priv-house-serv',
#  'occupation_ Prof-specialty', 'occupation_ Protective-serv',
#  'occupation_ Sales', 'occupation_ Tech-support',
#  'occupation_ Transport-moving',
#  'income_ <=50K', 'income_ >50K']

# 将data_dummies 数据转换为Numpy数组
features = data_dummies.loc[:, 'age': 'occupation_ Transport-moving']
# 提取Numpy数组
X = features.values
y = data_dummies['income_ >50K'].values
print('X.shape:{} y.shape:{}'.format(X.shape, y.shape))
# X.shape:(32561, 44) y.shape:(32561,)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Test score: {:.2f}'.format(logreg.score(X_test, y_test)))
# Test score: 0.81

# 数字可以编码分类变量
# 创建一个DataFrame，包含一个整数特征和一个分类字符串特征
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature' : ['socks', 'fox', 'socks', 'box']})
print(demo_df)
#           Integer Feature        Categorical Feature
# 0                0               socks
# 1                1                 fox
# 2                2               socks
# 3                1                 box

# 使用get_dummies只会编码字符串特征，不会改变整数特征
print(pd.get_dummies(demo_df))
#    Integer Feature  ...  Categorical Feature_socks
# 0                0  ...                          1
# 1                1  ...                          0
# 2                2  ...                          1
# 3                1  ...                          0

# 如果想为整数特征创建虚拟变量
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
tem = pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
print(tem)
#    Integer Feature_0  ...  Categorical Feature_socks
# 0                  1  ...                          1
# 1                  0  ...                          0
# 2                  0  ...                          1
# 3                  0  ...                          0
#
# [4 rows x 6 columns]

# 分箱、离散化、线性模型与树
# 数据表示的最佳方法不仅取决于数据的语义， 还取决于所使用的模型种类
X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label='linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='best')
plt.show()

# 特征分箱（离散化）可以让线性模型在连续数据上变得更加强大
# 将特征输入范围划分成固定个数的箱子（bin）
bins = np.linspace(-3, 3, 11)
print('bins:{}'.format(bins))
# bins:[-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]
# 记录每个数据点所属的箱子
# 这里做的是将数据集中单个连续输入特征变换为一个分类特征
which_bin = np.digitize(X, bins=bins)
print('\nData points:\n', X[:5])
print('\nBin membership for data points:\n', which_bin[:5])
#  Data points:
#  [[-0.75275929]
#  [ 2.70428584]
#  [ 1.39196365]
#  [ 0.59195091]
#  [-2.06388816]]
#
# Bin membership for data points:
#  [[ 4]
#  [10]
#  [ 8]
#  [ 6]
#  [ 2]]

# 将离散特征变换为one-hot编码
encoder = OneHotEncoder(sparse=False)
# encoder.fit找到which_bin中的唯一值
encoder.fit(which_bin)
# transform创建one-hot编码
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# 由于指定了10个箱子，所以变换后的数据集有10个特征
print('X_binned.shape:{}'.format(X_binned.shape))
# X_binned.shape:(100, 10)

# 对one-hot编码后的数据上构造新的线性模型和决策树模型
line_binned = encoder.transform(np.digitize(line, bins=bins))
print('{}\nshape:{}'.format(line_binned, line_binned.shape))
# [[1. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 1.]]
# shape:(1000, 10)

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.show()
# 发现线性回归和决策树做出了相似的预测，对于每个箱子，预测值都相同
# 分箱后对于线性模型更灵活，而基于树的模型通常不会产生更好的效果
# 分箱通常针对的是单个特征

# 交互特征与多项式特征
# 用于统计建模，也常用于许多实际的机器学习应用中
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)
# (100, 11)

reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression combined')
plt.legend(loc='best')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()
# 可以看到每个箱子的斜率都相同
# 希望每个箱子有不同斜率，可以添加交互特征或特征乘积
# 特征乘积
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
# (100, 20)

reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='best')
plt.show()

# 使用原始特征的多项式扩展连续特征
# 包含直到x ** 10的多项式
# 默认的'include_bias=True'添加恒等于1的常数特征
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
# 多项式的次数为10， 因此生成了10个特征
print('X_poly.shape:{}'.format(X_poly.shape))
# X_poly.shape:(100, 10)

# 比较X和X_poly
print('Entries of X:\n{}'.format(X[:5]))
print('Entries of X_poly:\n{}'.format(X_poly[:5]))
# Entries of X:
# [[-0.75275929]
#  [ 2.70428584]
#  [ 1.39196365]
#  [ 0.59195091]
#  [-2.06388816]]
# Entries of X_poly:
# [[-7.52759287e-01  5.66646544e-01 -4.26548448e-01  3.21088306e-01
#   -2.41702204e-01  1.81943579e-01 -1.36959719e-01  1.03097700e-01
#   -7.76077513e-02  5.84199555e-02]
#  [ 2.70428584e+00  7.31316190e+00  1.97768801e+01  5.34823369e+01
#    1.44631526e+02  3.91124988e+02  1.05771377e+03  2.86036036e+03
#    7.73523202e+03  2.09182784e+04]
#  [ 1.39196365e+00  1.93756281e+00  2.69701700e+00  3.75414962e+00
#    5.22563982e+00  7.27390068e+00  1.01250053e+01  1.40936394e+01
#    1.96178338e+01  2.73073115e+01]
#  [ 5.91950905e-01  3.50405874e-01  2.07423074e-01  1.22784277e-01
#    7.26822637e-02  4.30243318e-02  2.54682921e-02  1.50759786e-02
#    8.92423917e-03  5.28271146e-03]
#  [-2.06388816e+00  4.25963433e+00 -8.79140884e+00  1.81444846e+01
#   -3.74481869e+01  7.72888694e+01 -1.59515582e+02  3.29222321e+02
#   -6.79478050e+02  1.40236670e+03]]

# 通过调用方法查看特征语义
print('Polynomial feature names:\n{}'.format(poly.get_feature_names()))
# Polynomial feature names:
# ['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']

# 将多项式特征与线性回归模型一起使用， 可以得到经典的多项式回归模型
reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.ylabel('Input feature')
plt.legend(loc='best')
plt.show()

# 作为对比，在原始数据上学习核SVM模型
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma = {}'.format(gamma))
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='best')
plt.show()

# 在波士顿房价数据上做测试
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

# 缩放数据
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 提取多项式特征和交互特征，次数最高为2
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print('X_train.shape:{}'.format(X_train.shape))
print('X_train_poly.shape:{}'.format(X_train_poly.shape))
print('Polynomial feature names:\n{}'.format(poly.get_feature_names()))
# X_train.shape:(379, 13)
# X_train_poly.shape:(379, 105)

# 我们对Ridge回归在有交互特征数据上和没有交互特征的数据上的性能进行对比
ridge = Ridge().fit(X_train_scaled, y_train)
print('Score without interactions:{:.3f}'.format(
    ridge.score(X_test_scaled, y_test)
))

ridge = Ridge().fit(X_train_poly, y_train)
print('Score with interactions:{:.3f}'.format(ridge.score(X_test_poly, y_test)))
# Score without interactions:0.577
# Score with interactions:0.741
# 效果得到提升
# 如果使用更复杂的模型（如随机森林）效果不一定有提升
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print('Score without interactions:{:.3f}'.format(
    rf.score(X_test_scaled, y_test)
))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print('Score with interactions:{:.3f}'.format(rf.score(X_test_poly, y_test)))
# Score without interactions:0.770
# Score with interactions:0.764

# 单变量非线性变量
# 大部分模型都在每个特征大致遵循高斯分布时表现最好
# 使用模拟的技术数据集，特征全都是整数，响应都是连续的
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

# 计算第一个特征的前10个元素出现的次数
print('Number of feature appearances:\n{}'.format(np.bincount(X[:, 0])))
# Number of feature appearances:
# [28 38 68 48 61 59 45 56 37 40 35 34 36 26 23 26 27 21 23 23 18 21 10  9
#  17  9  7 14 12  7  3  8  4  5  5  3  4  2  4  1  1  3  2  5  3  8  2  5
#   2  1  2  3  3  2  2  3  3  0  1  2  1  0  0  3  1  0  0  0  1  3  0  1
#   0  2  0  1  1  0  0  0  0  1  0  0  2  2  0  1  1  0  0  0  0  1  1  0
#   0  0  0  0  0  0  1  0  0  0  0  0  1  1  0  0  1  0  0  0  0  0  0  0
#   1  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]

# 可视化看得更明白
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='r')
plt.ylabel('Number of appearances')
plt.xlabel('Value')
plt.show()
# 大多数模型无法很好地处理这些数据，如下，我们尝试拟合一个岭回归模型
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print('Test score:{:.3f}'.format(score))
# Test score:0.622

# 应用对数变换将数据分布更均匀
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel('Number of appearances')
plt.xlabel('Value')
plt.show()

# 在新数据集上构建岭回归，可以得到更好的拟合
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print('Test score:{:.3f}'.format(score))
# Test score:0.875

# 为数据集和模型的所有组合寻找最佳变换，在某种程度上是一种艺术
# 基于树的模型常常能够自己发现重要的交互项

# 自动化特征选择
# 如何判断每个特征的作用有多大？
# 单变量统计
# 基于模型的选择
# 迭代选择

# 单变量统计
# 单变量测试的计算速度很快，且不需要构建模型，完全独立于模型
# 要在scikit-learn中使用单变量特征选择，对于回归问题使用f_regression,对于分类问题使用f_classif
# 测试后获得p值，使用阈值来舍弃所有p值较大的特征

# 将分类的特征选择用于cancer数据集
# 为增加难度，向数据集中添加噪声特征，期望特征选择可以删除它们
cancer = load_breast_cancer()
# 获得确定性的随机数
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# 向数据集中添加噪声特征
# 前30个特征来自数据集，后50个是噪声
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5
)
# 使用f_classif(默认值)和Selectpercentile来选择50％的特征
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# 对训练集进行变换
X_train_select = select.transform(X_train)
print('X_train.shape:{}'.format(X_train.shape))
print('X_train_select.shape:{}'.format(X_train_select.shape))
# X_train.shape:(284, 80)
# X_train_select.shape:(284, 40)

# 可以调用方法查看哪些特征被选中
mask = select.get_support()
print(mask)
# [ True  True  True  True  True  True  True  True  True False  True False
#   True  True  True  True  True  True False False  True  True  True  True
#   True  True  True  True  True  True False False False  True False  True
#  False False  True False False False False  True False False  True False
#  False  True False  True False False False False False False  True False
#   True False False False False  True False  True False False False False
#   True  True False  True False False False False]

# 将遮罩可视化--黑色为True, 白色为False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample index')
plt.show()

# 比较Logistic回归在所有特征上的性能与仅使用所选特征的性能
# 对测试数据进行变换
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print('Score with all features:{:.3f}'.format(lr.score(X_test, y_test)))
lr = LogisticRegression()
lr.fit(X_train_select, y_train)
print('Score with only selected features:{:.3f}'.format(
    lr.score(X_test_selected, y_test)
))
# Score with all features:0.919
# Score with only selected features:0.919

# 基于模型的特征选择
# 与单变量选择不同，基于模型的选择同时考虑所有特征,由此可以获取交互项
# SelectFromModel类选出重要性度量（由监督模型提供）大于给定阈值的所有特征
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'
)
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print('X_trian.shape:{}'.format(X_train.shape))
print('X_train_l1.shape:{}'.format(X_train_l1.shape))
# X_trian.shape:(284, 80)
# X_train_l1.shape:(284, 40)

# 查看选中的特征
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample index')
plt.show()

# 查看性能
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print('Test score:{:.3f}'.format(score))
# Test score:0.930

# 迭代特征选择
# 在迭代特征选择中，将会构建一系列模型，每个模型都使用不同数量的特征，逐个增加或逐个减少
# 有一种特殊方法是递归特征消除（RFE）
# select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
#              n_features_to_select=10)
# select.fit(X_train, y_train)
# # 将选中的特征可视化
# mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel('Sample index')
# # 效果好于单变量、基于模型的特征选择
# # 查看回归模型精度
# X_train_ref = select.transform(X_train)
# X_test_ref = select.transform(X_test)
#
# score = LogisticRegression().fit(X_train_ref, y_train).score(X_test_ref, y_test)
# print('Test score:{:.3f}'.format(score))
# Test score:0.951
# 在RFE的内部使用随机森林的性能，只要选择了正确的特征，线性模型的变现就与随机森林一样好

# 利用专家知识
# 预测在Andreas家门口的自行车出租
# 将这个站点2015年8月的数据加载为pandas数据框，我们将数据重新采集为每3小时一个数据，以得道每一天的主要趋势
citibike = mglearn.datasets.load_citibike()
print('citibike shape:{}'.format(citibike.shape))
# citibike shape:(248,)
print('Citi Bike data:\n{}'.format(citibike.head()))
# Citi Bike data:
# starttime
# 2015-08-01 00:00:00     3
# 2015-08-01 03:00:00     0
# 2015-08-01 06:00:00     9
# 2015-08-01 09:00:00    41
# 2015-08-01 12:00:00    39

# 给出整个月出租车数量的可视化
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                       freq='D')
plt.xticks(xticks, xticks.strftime('%a %m-%d'), rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel('Data')
plt.ylabel('Rentals')
plt.show()
# 我们想根据过去预测未来
# 计算机上存储日期的方式是使用POSIX时间（以1970年为起点）计算秒值
# 使用单一整数特征（这里为秒值）作为数据表示
# 首先提取目标值（租车数量）
y = citibike.values
# 将时间转换为POSIX时间
X = citibike.index.astype("int64").values.reshape(-1, 1)


# 首先定义一个函数可以将数据集和测试集分开，构建模型并将结果可视化
# 使用184个数据点用于训练，剩余的数据用于测试
n_train = 184
# 对给定特征集上的回归进行评估和作图的函数


def eval_on_features(features, target, regressor):
    # 将给定特征划分为训练集和测试集
    X_train, X_test = features[:n_train], features[n_train:]
    # 同样划分目标数组
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print('Test-set R^2:{:.2f}'.format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime('%a %m-%d'), rotation=90,
               ha='left')
    plt.plot(range(n_train), y_train, label='train')
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label='prediction train')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label='prediction test')
    plt.legend(loc=(1.01, 0))
    plt.xlabel('Data')
    plt.ylabel('Rentals')


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)
plt.show()
# Test-set R^2:-0.04

# 测试集上的预测是一条直线，显然不正确
# 问题在于特征和随机森林的组合，测试集中时间戳特征值显然晚于训练集中所有点
# 树及随机森林无法外推到训练集之外的特征范围
# 利用专家知识
# 两个因素似乎非常重要：一天内的时间与一周的星期几，故我们添加这两个特征
# 我们从时间戳中学不到东西，故删除这个特征
# 首先仅使用每天的时刻
X_hour = citibike.index.hour.astype("int64").values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)
plt.show()
# Test-set R^2:0.60
# 预测的结果每天都要相同的模式

# 我们添加一周的星期几作为特征
X_hour_week = np.hstack([citibike.index.dayofweek.astype('int64').values.reshape(-1, 1),
                         X_hour])
eval_on_features(X_hour_week, y, regressor)
plt.show()
# Test-set R^2:0.84

# 使用线性回归进行测试
eval_on_features(X_hour_week, y, LinearRegression())
plt.show()
# Test-set R^2:0.13
# 线性模型较为简单，一周内的时间被解释为连续变量，线性模型只学到每天时间的线性函数
# 我们可以将整数解释为分类变量
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
eval_on_features(X_hour_week_onehot, y, Ridge())
plt.show()
# 效果得到提升，为每一天内的每一个时刻都学到了一个系数
# 利用交互特征，可以为星期几和时刻的每一种组合学到一个系数
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)
plt.show()
# 效果非常好
# Test-set R^2:0.85
# 将模型学到的系数作图，随机森林无法做到这点
hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour

# 利用get_feature_names方法对PolynomialFeatures提取的所有交互特征进行命名，并仅保留系数不为零的特征
features_poly = poly_transformer.get_feature_names(features)
print(features)
print(features_poly)
# ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', '00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']


# ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', '00:00',
# '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00',
# 'Mon Tue', 'Mon Wed', 'Mon Thu', 'Mon Fri',
# 'Mon Sat', 'Mon Sun', 'Mon 00:00', 'Mon 03:00',
# 'Mon 06:00', 'Mon 09:00', 'Mon 12:00', 'Mon 15:00', 'Mon 18:00',
# 'Mon 21:00', 'Tue Wed', 'Tue Thu', 'Tue Fri', 'Tue Sat', 'Tue Sun',
# 'Tue 00:00', 'Tue 03:00', 'Tue 06:00', 'Tue 09:00', 'Tue 12:00',
# 'Tue 15:00', 'Tue 18:00', 'Tue 21:00', 'Wed Thu', 'Wed Fri', 'Wed Sat',
# 'Wed Sun', 'Wed 00:00', 'Wed 03:00', 'Wed 06:00', 'Wed 09:00', 'Wed 12:00',
# 'Wed 15:00', 'Wed 18:00', 'Wed 21:00', 'Thu Fri', 'Thu Sat', 'Thu Sun',
# 'Thu 00:00', 'Thu 03:00', 'Thu 06:00', 'Thu 09:00', 'Thu 12:00',
# 'Thu 15:00', 'Thu 18:00', 'Thu 21:00', 'Fri Sat', 'Fri Sun',
# 'Fri 00:00', 'Fri 03:00', 'Fri 06:00', 'Fri 09:00', 'Fri 12:00',
# 'Fri 15:00', 'Fri 18:00', 'Fri 21:00', 'Sat Sun', 'Sat 00:00',
# 'Sat 03:00', 'Sat 06:00', 'Sat 09:00', 'Sat 12:00', 'Sat 15:00',
# 'Sat 18:00', 'Sat 21:00', 'Sun 00:00', 'Sun 03:00', 'Sun 06:00',
# 'Sun 09:00', 'Sun 12:00', 'Sun 15:00', 'Sun 18:00', 'Sun 21:00',
# '00:00 03:00', '00:00 06:00', '00:00 09:00', '00:00 12:00', '00:00 15:00',
# '00:00 18:00', '00:00 21:00', '03:00 06:00', '03:00 09:00', '03:00 12:00',
# '03:00 15:00', '03:00 18:00', '03:00 21:00', '06:00 09:00', '06:00 12:00',
# '06:00 15:00', '06:00 18:00', '06:00 21:00', '09:00 12:00', '09:00 15:00',
# '09:00 18:00', '09:00 21:00', '12:00 15:00', '12:00 18:00', '12:00 21:00',
# '15:00 18:00', '15:00 21:00', '18:00 21:00']

# 保留系数不为0的点
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

# 将线性模型学到的系数可视化
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel('Feature name')
plt.ylabel('Feature magnitude')
plt.show()








