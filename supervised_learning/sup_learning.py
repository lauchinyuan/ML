import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris
import mglearn
import graphviz
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # k近邻分类
from sklearn.neighbors import KNeighborsRegressor  # k近邻回归

# 线性模型
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.linear_model import Ridge  # 岭回归
from sklearn.linear_model import Lasso  # Lasso回归
# 线性模型用于分类
from sklearn.linear_model import LogisticRegression  # Logistic回归
from sklearn.svm import LinearSVC  # 线性支持向量机
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升回归树
from sklearn.svm import SVC  # 核支持向量机
from sklearn.neural_network import MLPClassifier  # 多层感知机
warnings.filterwarnings("ignore")

cancer_data = load_breast_cancer()
print('dict_keys:\n{}'.format(cancer_data.keys()))
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print('shape of data:{}'.format(cancer_data['data'].shape))
# shape of data:(569, 30)
# 统计数据Target情况
print('Sample count per class:\n{}'.format({n: v for n, v in zip(cancer_data['target_names'],
                                                                 np.bincount(cancer_data['target']))}))
# {'malignant': 212, 'benign': 357}
# 查看属性名称
print('Feature names:\n{}'.format(cancer_data['feature_names']))
# Feature names:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']


boston_data = load_boston()
print('Data shape:{}'.format(boston_data['data'].shape))
# Data shape:(506, 13)
# 扩展工程
X, y = mglearn.datasets.load_extended_boston()
print('Data shape:{}'.format(X.shape))
# Data shape:(506, 104)


# K近邻
# mglearn.plots.plot_knn_classification(n_neighbors=1)  # knn图像测试
# plt.show()

X, y = mglearn.datasets.make_forge()  # 获取数据

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # 分离数据

knn_clf = KNeighborsClassifier(n_neighbors=3)  # 实例化

knn_clf.fit(X_train, y_train)  # 训练分类模型

# 预测
print('Test set predictions{}'.format(knn_clf.predict(X_test)))
# Test set predictions[1 0 1 0 1 0 0]

# 评估模型
print('Test set accuracy:{}'.format(knn_clf.score(X_test, y_test)))
# Test set accuracy:0.8571428571428571

# 决策边界可视化
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# for n_neighbors, ax in zip([1, 3, 9], axes):  # 分别获取三个子图和三个k值
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(knn, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title('{} neighbor(s)'.format(n_neighbors))
#     ax.set_xlabel('feature0')
#     ax.set_ylabel('feature1')
# axes[0].legend(loc=3)
# plt.show()

# 模型复杂度和泛化能力的关系
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, random_state=66)
train_accuracy = []
test_accuracy = []
# n_neighbor取值从1到10
n_neighbors_setting = range(1, 11)
for n_neighbors in n_neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 记录训练精度
    train_accuracy.append(clf.score(X_train, y_train))
    # 记录泛化精度
    test_accuracy.append(clf.score(X_test, y_test))
# plt.plot(n_neighbors_setting, train_accuracy, label='training accuracy')
# plt.plot(n_neighbors_setting, test_accuracy, label='test accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('n_neighbors')
# plt.legend()
# plt.show()

# k近邻回归
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_test, y_test)
# 预测
print('Test set predictions:\n{}'.format(reg.predict(X_test)))
# [-0.63990497  0.54043149  0.65456667 -1.08307584 -1.1346844  -0.63990497
#   0.0377314   0.65456667  0.08946785 -1.1346844 ]

# 查看决定系数
print('Test set R^2:{}'.format(reg.score(X_test, y_test)))
# Test set R^2:0.6952306950424726

# # 分析KNeighborsRegressor
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# # 创建1000个数据点
# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     reg = KNeighborsRegressor(n_neighbors=n_neighbors)
#     reg.fit(X_train, y_train)
#     ax.plot(line, reg.predict(line))
#     ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
#     ax.plot(X_test, y_test, '*', c=mglearn.cm2(1), markersize=8)
#     ax.set_title('{}neighbor(s)\ntrain score:{:.2f} test score:{:.2f}'.format(
#         n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)
#     ))
# axes[0].legend(['Model predictions', 'Training data/target', 'Test data/target'], loc='best')
# plt.show()

# 线性模型
# mglearn.plots.plot_linear_regression_wave()
# plt.show()
# 线性回归
# 线性回归又称普通最小二乘法(OSL)
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print('coef:{}'.format(lr.coef_))  # 查看系数（coefficient）
# coef:[0.39390555]
print('intercept:{}'.format(lr.intercept_))
# intercept:-0.031804343026759746
# 查看训练集和测试集的性能
print('Training set score:{:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lr.score(X_test, y_test)))
# Training set score:0.67
# Test set score:0.66

# 波士顿房价数据实例
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print('Training set score:{:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lr.score(X_test, y_test)))
# Training set score:0.95
# Test set score:0.61  可以初步判断为过拟合

# 岭回归
# 正则化是指对模型作显式约束，以免过拟合
# 岭回归使用L2正则化
ridge = Ridge().fit(X_train, y_train)
print('Training set score:{:.2f}'.format(ridge.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(ridge.score(X_test, y_test)))
# Training set score:0.89
# Test set score:0.75

# 增大alpha可以使得系数更加趋近于0
ridge = Ridge(alpha=10).fit(X_train, y_train)
print('Training set score:{:.2f}'.format(ridge.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(ridge.score(X_test, y_test)))
# Training set score:0.79
# Test set score:0.64

# lasso回归
# lasso回归模型使用L1正则化方法，会令某些系数刚好为0
lasso = Lasso().fit(X_train, y_train)
print('Training set score:{:.2f}'.format(lasso.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lasso.score(X_test, y_test)))
print('Numbers of features used:{}'.format(np.sum(lasso.coef_ != 0)))
# Training set score:0.29
# Test set score:0.21
# Numbers of features used:4

# 降低欠拟合
# 设定alpha, 同时增大max_iter的值，否则模型会警告
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print('Training set score:{:.2f}'.format(lasso001.score(X_train, y_train)))
print('Test set score:{:.2f}'.format(lasso001.score(X_test, y_test)))
print('Numbers of features used:{}'.format(np.sum(lasso001.coef_ != 0)))
# Training set score:0.90
# Test set score:0.77
# Numbers of features used:33

# 用于分类的线性模型
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title('{}'.format(clf.__class__.__name__))
# axes[0].legend()
# plt.show()
# SVC模型和Logistic模型都使用L2正则化

# C值调整
# C值越大，正则化越弱
# 较少的C值可以让算法尽量适应大多数数据点，较大C值强调每个数据点分类正确的重要性

# LogisticRegression 分析乳腺癌数据
cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data['data'], cancer_data['target'],
                                                    random_state=42,  # stratify表示按指定数据划分
                                                    stratify=cancer_data['target'])
logreg = LogisticRegression().fit(X_train, y_train)
print('Training set score:{:.3f}'.format(logreg.score(X_train, y_train)))
print('Test set score:{:.3f}'.format(logreg.score(X_test, y_test)))
# Training set score:0.948
# Test set score:0.951
# Test、Training测试性能接近，可能欠拟合，故增大C值

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print('Training set score:{:.3f}'.format(logreg100.score(X_train, y_train)))
print('Test set score:{:.3f}'.format(logreg100.score(X_test, y_test)))
# Training set score:0.946
# Test set score:0.958

# 研究正则化更强时会发生什么
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print('Training set score:{:.3f}'.format(logreg001.score(X_train, y_train)))
print('Test set score:{:.3f}'.format(logreg001.score(X_test, y_test)))
# Training set score:0.934
# Test set score:0.930

# 对比正则化参数不同时，模型学习到的系数
# plt.plot(logreg.coef_.T, 'o', label='C=1')
# plt.plot(logreg100.coef_.T, '^', label='C=100')
# plt.plot(logreg001.coef_.T, '*', label='C=0.01')
# plt.xticks(range(cancer_data['data'].shape[1]), cancer_data['feature_names'], rotation=90)
# plt.ylim(-2, 2)
# plt.legend()
# plt.show()

# 用于多分类的线性模型
# 一对其余方法（one-vs-rest）, 每对类别都有一个系数向量和一个截距
X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.legend(['Class0', 'Class1', 'Class2'])
# plt.show()

# 在这个数据集上训练LinearSVC分类器
linear_svc = LinearSVC().fit(X, y)
print('Coefficient shape:{}'.format(linear_svc.coef_.shape))
print('Intercept shape:{}'.format(linear_svc.intercept_.shape))
# Coefficient shape:(3, 2)  此处3表示每种类别都有系数向量，2表示2个特征
# Intercept shape:(3,)  3个截距

# 将3个二类分类器给出的直线可视化
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_,
#                                   ['b', 'r', 'g']):
#     plt.plot(line, -(line * coef[0] + intercept)/coef[1], c=color)
#     plt.ylim(-10, 15)
#     plt.xlim(-10, 8)
#     plt.xlabel('Feature 0')
#     plt.ylabel('Feature 1')
#     plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc='best')
# plt.show()
# 给出二维空间中所有区域的预测结果
# mglearn.plots.plot_2d_classification(linear_svc, X, fill=True, alpha=.7)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_,
#                                   ['b', 'g', 'r']):
#     plt.plot(line, -(line * coef[0] + intercept)/coef[1], c=color)
#     plt.legend(['Class0', 'Class1', 'Class2', 'Line class0',
#                 'Line class 1', 'Line class 2'])
#     plt.xlabel('Feature 0')
#     plt.ylabel('Feature 1')
# plt.show()

# 朴素贝叶斯分类器
# 朴素贝叶斯分类器效率高， 泛化能力比线性分类器稍差
# 朴素贝叶斯分为GaussianNB, BernoulliNB, MultinomialNB

# 理解BernoulliNB分类器
# 计算每个类别中每个特征不为0的元素个数
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}  # 创建空字典
for label in np.unique(y):
    # 对每个类别进行遍历
    # 计算每个特征中1的个数
    counts[label] = X[y == label].sum(axis=0)
print('Feature counts:\n{}'.format(counts))
# {0: array([0, 1, 0, 2]), 1: array([2, 0, 2, 1])}
# 其他两种朴素贝叶斯原理相似
# GaussianNB主要用于高维数据
# 朴素贝叶斯常用于大型数据集，鲁棒性较好，是很好的基准模型

# 决策树
# 构造决策树
# 防止决策树过拟合可以采用预剪枝或后剪枝
# sklearn只实现预剪枝
cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data['data'], cancer_data['target'],
                                                    random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set:{:3f}'.format(tree.score(X_test, y_test)))
# Accuracy on training set:1.000
# Accuracy on test set:0.930070

# 限制树的深度
tree = DecisionTreeClassifier(random_state=0, max_depth=4)
tree.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set:{:3f}'.format(tree.score(X_test, y_test)))
# Accuracy on training set:0.995
# Accuracy on test set:0.951049

# 分析决策树
# 查看特征重要性
print('Feature importance:\n{}'.format(tree.feature_importances_))
# [0.         0.         0.         0.         0.         0.
#  0.         0.73943775 0.         0.         0.013032   0.
#  0.         0.         0.         0.         0.         0.01737208
#  0.00684355 0.         0.06019401 0.11783988 0.         0.03522339
#  0.01005736 0.         0.         0.         0.         0.        ]


# 可视化特征重要性
def plot_feature_importance_cancer(model):
    n_features = cancer_data['data'].shape[1]  # 得到特征数
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(range(n_features), cancer_data['feature_names'])
    plt.xlabel('Feature importance')
    plt.ylabel('Features')
    plt.show()


# plot_feature_importance_cancer(tree)

# 决策树用于回归
# 决策树回归不能外推，也不能在训练数据以外进行预测
# 决策树即使做了预剪枝，也常常会过拟合， 泛化性能很差

# 决策树集成

# 随机森林
# 随机森林中，每棵树都可能对部分数据过拟合，不同的树以不同的方式过拟合，对结果取平均可以降低过拟合
# 随机森林能减少过拟合又能保持树的预测能力
# 随机森林构造时采用自助采样，且使用特征子集，保证了随机森林每棵树都不同
# 随机森林中max_features太大会导致随机森林中的树十分相似
# 随机森林中max_features太小会导致树差异较大，需要较大的深度才能预测
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)  # 生成5棵树
forest.fit(X_train, y_train)
# 树被保存在estimator_中
# 将每棵树学到的决策边界以及总的决策边界可视化
# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     ax.set_title('Tree {}'.format(i))
#     mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
# axes[-1, -1].set_title('Random Forest')
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.show()


# 使用100棵树
X_train, X_test, y_train, y_test = train_test_split(cancer_data['data'], cancer_data['target'],
                                                    random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(forest.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(forest.score(X_test, y_test)))
# Accuracy on training set:1.000
# Accuracy on test set:0.972

# 随机森林的特征重要性
# plot_feature_importance_cancer(forest)
# 随机森林在维度很高的稀疏数据中表现不是很好
# 随机森林深度一般比树深
# n_estimators在时间和内存允许的情况下尽量多


# 梯度提升回归树（梯度提升机）
# 梯度提升采用连续方法构造树，每棵树都试图纠正前一棵树的错误
# 使用强预剪枝，深度小，占用内存小，速度快
# 学习率越高、模型越复杂
X_train, X_test, y_train, y_test = train_test_split(
    cancer_data['data'], cancer_data['target'], random_state=0
)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(gbrt.score(X_test, y_test)))
# Accuracy on training set:1.000
# Accuracy on test set:0.965

# 可能存在过拟合，降低学习率，或者限制最大深度
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(gbrt.score(X_test, y_test)))
# Accuracy on training set:0.991
# Accuracy on test set:0.972

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(gbrt.score(X_test, y_test)))
# Accuracy on training set:0.988
# Accuracy on test set:0.965

# 核支持向量机
# 线性模型在低维空间变现受限（线和平面的灵活性有限）
# 对输入特征进行扩展
# 添加第二个特征的平方，作为一个新特征
X_new = np.hstack([X, X[:, 1:] ** 2])
figure = plt.figure()
# 3D可视化
# ax = Axes3D(figure, elev=-152, azim=-26)
# # 首先画所有y==0的点，然后画所有y==1的点
# mask = y == 0
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
#            cmap=mglearn.cm2, s=60)
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',
#            cmap=mglearn.cm2, s=60)
# ax.set_xlabel('Feature 0')
# ax.set_ylabel('Feature 1')
# ax.set_zlabel('Feature 1 ** 2')
# plt.show()

# 用线性模型拟合扩展后的数据
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# 显示线性决策边界
# figure = plt.figure()
# ax = Axes3D(figure, elev=-152, azim=-26)
# xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)  # x坐标
# yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
#
# XX, YY = np.meshgrid(xx, yy)
# ZZ = (coef[0] * XX + coef[1] * YY + intercept)/-coef[2]
# ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
#            cmap=mglearn.cm2, s=60)
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',
#            cmap=mglearn.cm2, s=60)
# ax.set_xlabel('Feature 0')
# ax.set_ylabel('Feature 1')
# ax.set_zlabel('Feature 1 ** 2')
# plt.show()

# 核技巧直接计算扩展特征表中的数据点中间的距离，而不用实际对扩展进行计算
# 理解SVM
# 通常只有一部分训练数据点对于定义决策边界来说很重要
# 位于类别边界上的点称为支持向量
# 支持向量重要性
# 高斯核公式
# X, y = mglearn.tools.make_handcrafted_dataset()
# svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)  # rbf表示高斯核
# mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # 画出支持向量
# sv = svm.support_vectors_
# # 支持向量的类别标签由dual_coef_的正负号给出
# sv_labels = svm.dual_coef_.ravel() > 0
# mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.show()

# SVM调参
# gamma参数控制高斯核密度，决定点与点之间的"靠近"是指多少距离
# gamma较小说明高斯核半径较大，决策边界变化较慢，模型复杂度较低
# C值小时，说明每个数据点的影响范围有限
# 默认情况下， C = 1， gamma = 1/n_features
# fig, axes = plt.subplots(3, 3, figsize=(15, 10))
# for ax, C in zip(axes, [-1, 0, 3]):
#     for a, gamma in zip(ax, range(-1, 2)):
#         mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
# axes[0, 0].legend(['class 0', 'class1', 'sv class 0', 'sv class 1'], ncol=4, loc=(.9, 1.2))


# 将RBF核SVM应用到乳腺癌数据集上
cancer_data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data['data'],
                                                    cancer_data['target'], random_state=0)
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
print('Accuracy on training set:{:.3f}'.format(svc.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(svc.score(X_test, y_test)))
# Accuracy on training set:0.988
# Accuracy on test set:0.965

# 对于核SVM模型来说特征变化范围有较大影响
# plt.plot(X_train.min(axis=0), 'o', label='min')
# plt.plot(X_train.max(axis=0), '^', label='max')
# plt.legend(loc=4)
# plt.xlabel('Feature index')
# plt.ylabel('Feature magnitude')
# plt.yscale('log')
# plt.show()
# 可以看到数据集特征具有不同数量级

# 处理问题：为SVM预处理数据
# 计算训练集中每个特征的最小值
min_on_training = X_train.min(axis=0)
# 计算训练集中每个特征的范围
range_on_training = (X_train - min_on_training).max(axis=0)

# 减去最小值，然后除以范围，得到的特征最小值即为0， 最大值即为1
X_train_scaled = (X_train - min_on_training)/range_on_training
print('Minimum of each feature\n{}'.format(X_train_scaled.min(axis=0)))
print('Maximum of each feature\n{}'.format(X_train_scaled.max(axis=0)))
# Minimum of each feature
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]
# Maximum of each feature
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1.]

# 利用训练集最小值和范围对测试集做相同的变换
X_test_scaled = (X_test - min_on_training)/range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
print('Accuracy on training set:{:.3f}'.format(svc.score(X_train_scaled, y_train)))
print('Accuracy on test set:{:.3f}'.format(svc.score(X_test_scaled, y_test)))
# Accuracy on training set:0.984
# Accuracy on test set:0.972
# 提升效果好，但有欠拟合之虞，此时调参增大模型复杂度
svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print('Accuracy on training set:{:.3f}'.format(svc.score(X_train_scaled, y_train)))
print('Accuracy on test set:{:.3f}'.format(svc.score(X_test_scaled, y_test)))
# Accuracy on training set:1.000
# Accuracy on test set:0.958

# 核支持向量机在各种数据上都变现很好，但对样本数据的缩放不好
# 样本数据很多时，在内存和时间方面会面临挑战

# 神经网络（深度学习）
# 多层感知机（MLP）也称为普通前馈神经网络
# 非线性函数:校正非线性（relu）和正切双曲线（tanh）
# 两个非线性函数可视化
line = np.linspace(-3, 3, 100)
# plt.plot(line, np.tanh(line), label='tanh')
# plt.plot(line, np.maximum(line, 0), label='relu')
# plt.legend(loc='best')
# plt.xlabel('x')
# plt.ylabel('relu(x), tanh(x)')
# plt.show()

# 神经网络调参
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[50]).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 使用2个隐层，每个包含10个单元
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 改用tanh非线性
mlp = MLPClassifier(solver='lbfgs', random_state=0,
                    hidden_layer_sizes=[10, 10], activation='tanh').fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 显示不同alpha值对数据集的影响
# 两个隐层神经网络，每层使用10个或20个单元
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 20]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                            alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        ax.set_title('n_hidden = [{},{}]\nalpha={:.4f}'.format(
            n_hidden_nodes, n_hidden_nodes, alpha
        ))
plt.show()

# 不同的随机参数的影响
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs', random_state=i,
                        hidden_layer_sizes=[10, 10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
plt.show()
# 可以看到学习结果不同

# 将MLPClassifier用于乳腺癌数据集上
print('Cancer data per-feature maximum:\n{}'.format(cancer_data['data'].max(axis=0)))
# Cancer data per-feature maximum:
# [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 3.454e-01 4.268e-01
#  2.012e-01 3.040e-01 9.744e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
#  3.113e-02 1.354e-01 3.960e-01 5.279e-02 7.895e-02 2.984e-02 3.604e+01
#  4.954e+01 2.512e+02 4.254e+03 2.226e-01 1.058e+00 1.252e+00 2.910e-01
#  6.638e-01 2.075e-01]

X_train, X_test, y_train, y_test = train_test_split(
    cancer_data['data'], cancer_data['target'], random_state=0
)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print('Accuracy on training set:{:.2f}'.format(mlp.score(X_train, y_train)))
print('Accuracy on test set:{:.3f}'.format(mlp.score(X_test, y_test)))
# Accuracy on training set:0.94
# Accuracy on test set:0.916
# 神经网络也要求所有输入特征的变化范围相似，最理想的情况为均值为0，方差为1

# 计算训练集中每个特征的平均值
mean_on_training = X_train.mean(axis=0)
# 计算训练集中每个特征的标准差
std_on_tringing = X_train.std(axis=0)

# 减去平均值，然后乘以标准差的倒数
# 如此计算之后，mean = 0, std = 1
X_train_scaled = (X_train - mean_on_training)/std_on_tringing
# 对测试集作相同的变换（使用训练集的平均值和标准差）
X_test_scaled = (X_test - mean_on_training)/std_on_tringing
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print('Accuracy on training set:{:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on test set:{:.3f}'.format(mlp.score(X_test_scaled, y_test)))
# Accuracy on training set:0.991
# Accuracy on test set:0.965
# 可以看到结果好很多

# 增加迭代次数
mlp = MLPClassifier(random_state=0, max_iter=1000)
mlp.fit(X_train_scaled, y_train)
print('Accuracy on training set:{:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy on test set:{:.3f}'.format(mlp.score(X_test_scaled, y_test)))
# Accuracy on training set:1.000
# Accuracy on test set:0.972

# 隐层权重热图可视化
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer_data['feature_names'])
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()
plt.show()

# 分类器的不确定度估计
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
# 为了便于说明，将两个类重命名为'Blue'和'red'
y_named = np.array(['blue', 'red'])[y]

# 我们可以对任意个数组调用train_test_split()
# 对所有数组划分方式都是一致的
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(
    X, y_named, y, random_state=0
)
# 构建梯度提升模型
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# 决策函数
print('X_test.shape:{}'.format(X_test.shape))
print('Decision function shape:{}'.format(
    gbrt.decision_function(X_test).shape
))
# X_test.shape:(25, 2)
# Decision function shape:(25,)

# 显示Decision_function的前几个元素
print('Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6]))
# Decision function:
# [ 4.13592629 -1.7016989  -3.95106099 -3.62599351  4.28986668  3.66166106]

# 可以通过查看决策函数返回值的正负号来再现预测值
print('thresholded decision function:\n{}'.format(
    gbrt.decision_function(X_test) > 0))
print('Predictions:{}'.format(gbrt.predict(X_test)))
# thresholded decision function:
# [ True False False False  True  True False  True  True  True False  True
#   True False  True False False False  True  True  True  True  True False
#  False]
# Predictions:['red' 'blue' 'blue' 'blue' 'red' 'red' 'blue' 'red' 'red' 'red' 'blue'
#  'red' 'red' 'blue' 'red' 'blue' 'blue' 'blue' 'red' 'red' 'red' 'red'
#  'red' 'blue' 'blue']

# 查看decision_function的范围,由于可以缩放，通常难以解释
decision_function = gbrt.decision_function(X_test)
print('MAX:{}\nMIN:{}'.format(np.max(decision_function),
                              np.min(decision_function)))
# MAX:4.289866676868515
# MIN:-7.69097177301218

# 可视化取值
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                           alpha=.4, cm=mglearn.ReBl)
for ax in axes:
    # 画出训练点和测试点
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
cbar = plt.colorbar(score_image, ax=axes.tolist())
axes[0].legend(['Test class 0', 'Test class 1', 'Train class 0', 'Train class 1'],
               ncol=4, loc=(.1, 1.1))
plt.show()

# 预测概率
print('Shape of probabilities:{}'.format(gbrt.predict_proba(X_test).shape))
# Shape of probabilities:(25, 2)
# 第一个为元素类别1的概率，第二个为类别2的概率
# 显示Predict_proba的前几个元素
print('Predicted probabilities:\n{}'.format(gbrt.predict_proba(X_test[:6])))
# Predicted probabilities:
# [[0.01573626 0.98426374]
#  [0.84575649 0.15424351]
#  [0.98112869 0.01887131]
#  [0.97406775 0.02593225]
#  [0.01352142 0.98647858]
#  [0.02504637 0.97495363]]

# 预测概率可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
score_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                           alpha=.4, cm=mglearn.ReBl, function='predict_proba')
for ax in axes:
    # 画出训练点和测试点
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
cbar = plt.colorbar(score_image, ax=axes.tolist())
axes[0].legend(['Test class 0', 'Test class 1', 'Train class 0', 'Train class 1'],
               ncol=4, loc=(.1, 1.1))
plt.show()
# 比前面的图像更为突出

# 多分类问题的不确定度
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris['data'], iris['target'], random_state=42
)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=42)
gbrt.fit(X_train, y_train)
# 显示决策函数的前几个元素
print('Decision function shape:{}'.format(gbrt.decision_function(X_test).shape))
print('Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6]))
# [[-1.9957153   0.04758118 -1.92721297]
#  [ 0.0614655  -1.90755689 -1.92793177]
#  [-1.99075072 -1.87637856  0.09686741]
#  [-1.9957153   0.04758118 -1.92721297]
#  [-1.99730166 -0.13469231 -1.20341532]
#  [ 0.0614655  -1.90755689 -1.92793177]]

# 根据预测分数再现预测结果
print('Argmax of decision function:\n{}'.format(
    np.argmax(gbrt.decision_function(X_test), axis=1)
))
print('Predictions:\n{}'.format(gbrt.predict(X_test)))
# Argmax of decision function:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
# Predictions:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]

# predict_prob也有一样的效果
print('Agrmax of predicted probabilities:\n{}'.format(
    np.argmax(gbrt.predict_proba(X_test), axis=1)
))
print('Predictions:\n{}'.format(gbrt.predict(X_test)))
# Agrmax of predicted probabilities:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]
# Predictions:
# [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1
#  0]

# 获取分类器真实的属性名称
logreg = LogisticRegression()
named_target = iris['target_names'][y_train]  # 为目标命名
logreg.fit(X_train, named_target)
print('unique classes in training data:{}'.format(logreg.classes_))
print('redictions:{}'.format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print('argmax of decision function:{}'.format(argmax_dec_func[:10]))
print('argmax combined with classes_:{}'.format(
    logreg.classes_[argmax_dec_func][:10]
))

# unique classes in training data:['setosa' 'versicolor' 'virginica']
# redictions:['versicolor' 'setosa' 'virginica' 'versicolor' 'versicolor' 'setosa'
#  'versicolor' 'virginica' 'versicolor' 'versicolor']
# argmax of decision function:[1 0 2 1 1 0 1 2 1 1]
# argmax combined with classes_:['versicolor' 'setosa' 'virginica' 'versicolor' 'versicolor' 'setosa'
#  'versicolor' 'virginica' 'versicolor' 'versicolor']





