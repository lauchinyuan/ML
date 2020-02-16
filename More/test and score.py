import matplotlib
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score  # 交叉验证
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import SCORERS
# 交叉验证
# 交叉验证是评估泛化性能的统计学方法
# 最常见的交叉验证是K折交叉验证，其中k是用户指定的数字
# 可视化示意效果
mglearn.plots.plot_cross_validation()
plt.show()

# 在iris数据集上对LogisticRegression进行评估
iris = load_iris()
logreg = LogisticRegression()
score = cross_val_score(logreg, iris.data, iris.target)
print('Cross-validation scores:{}'.format(score))
# 默认情况下执行五折交叉验证
# Cross-validation scores:[0.96666667 1.         0.93333333 0.96666667 1.        ]
# 改变k值
score = cross_val_score(logreg, iris.data, iris.target, cv=3)
print('Cross-validation scores:{}'.format(score))
# Cross-validation scores:[0.98 0.96 0.98]

# 总结交叉验证的一种常用方法是计算平均值
print('Average cross-validation score:{:.2f}'.format(score.mean()))
# Average cross-validation score:0.97

# 分层k值交叉验证和其他策略
# 将数据集划分为k折时，可能并不是一个好主意，例如我们下面看到的数据
iris = load_iris()
print('Iris labels:\n{}'.format(iris.target))
#
# 显然以上数据不太适用于k折交叉验证
# 使用分层k折交叉验证使每个折中类别比例与数据集中的比例相同
mglearn.plots.plot_stratified_cross_validation()
plt.show()

# 对交叉验证的更多控制
# sklearn允许提供一个交叉验证分离器作为cv参数，来对数据划分过程进行更精细的控制
kfold = KFold(n_splits=5)
# 可以将kfold分离器对象作为cv参数传入cross_val_score
print('Cross-validation score:\n{}'.format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)
))
# [1.         1.         0.86666667 0.93333333 0.83333333]
# 我们可以验证在iris数据集上使用3折交叉验证（不分层）确实是一个非常糟糕的主意
kfold = KFold(n_splits=3)
print('Cross-validation score:\n{}'.format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)
))

# 学不到任何内容，另一种方法是可以将数据打乱来代替分层
kfold = KFold(n_splits=3, shuffle=True, random_state=0)  # shuffle参数可以打乱数据
print('Cross-validation score:\n{}'.format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)
))
# [0.98 0.96 0.96]

# 留一法交叉验证（LeaveOneOut）
# 留一法是交叉验证的特殊情况
# 此方法较费时，适用于小数据
loo = LeaveOneOut()
score = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print('Number of cv interation:', len(score))
print('Mean accuracy:{:.2f}'.format(score.mean()))
# Number of cv interation: 150
# Mean accuracy:0.97

# 打乱划分交叉验证（ShuffleSplit）
# 另一种灵活的交叉验证策略是打乱划分交叉验证
mglearn.plots.plot_shuffle_split()
plt.show()

# 将数据集划分为50％的训练集和测试集，共运行10次迭代
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
score = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print('Cross-validation score:\n{}'.format(score))
# Cross-validation score:
# [0.98666667 0.94666667 0.97333333 0.96       0.97333333 0.94666667
#  0.97333333 0.96       0.93333333 0.94666667]

# 分组交叉验证
# 适用于数据集中的分组高度相关时
# 例如人脸情绪识别中需要避免在数据集和测试集中出现同一个人的不同情绪，这会令预测结果偏好
# 我们想把每个人的不同情绪分为一组，而不会分散在测试集和数据集中
# 我们可以使用GroupKFold实现这点
# 下面的例子中包含12个数据点，共分为4个组
X, y = make_blobs(n_samples=12, random_state=0)
# 假设前3个样本属于同一组，接下来4个属于同一组，以此类推
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print('Cross-validation scores:\n{}'.format(scores))
# Cross-validation scores:
# [0.75       0.6        0.66666667]

# 可视化分组
mglearn.plots.plot_group_kfold()
plt.show()

# sklearn中还有很多交叉验证的划分策略，详细查看用户指南

# 网格搜索
# 自己实现简单网格搜索
X_train, X_test, y_train, y_test = train = train_test_split(
    iris.data, iris.target, random_state=0
)
print('Size of training set:{}\nSize of test set:{}'.format(
    X_train.shape[0], X_test.shape[0]
))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # 对每种参数都训练一个SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # 在测试集上评估SVC
        score = svm.score(X_test, y_test)
        # 如果我们得到更好的分数，则保存参数和分数
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
print('Best score:{:.2f}'.format(best_score))
print('Best parameters:{}'.format(best_parameters))
# Size of training set:112
# Size of test set:38
# Best score:0.97
# Best parameters:{'C': 100, 'gamma': 0.001}

# 参数过拟合的风险与验证集
# 我们使用了测试集进行调参，不能再用它来评估模型的好坏
# 为了解决这个问题，一种方法是再次划分数据，再分出验证集（开发集）来选择模型参数

# 如图
mglearn.plots.plot_threefold_split()
plt.show()

# 实例：将数据划分为训练集+验证集与测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
# 将训练+验证集划分为训练集与验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
print('Size of training set:{}\nSize of validation set:{}\nSize of test set:{}'.format(
    X_train.shape[0], X_valid.shape[0], X_test.shape[0]
))
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# 在训练+验证集上重新构建一个模型，并在测试集上进行评估
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print('Best score on validation set:{:.2f}'.format(best_score))
print('Best parameters:', best_parameters)
print('Test set score with best parameter:{:.2f}'.format(test_score))
# Best score on validation set:0.97
# Best parameters: {'C': 100, 'gamma': 0.001}
# Test set score with best parameter:0.97

# 带交叉验证的网格搜索
# 将数据划分为训练集、验证集和测试集的方法是可行的，但对数据的划分方式很敏感
# 我们可以使用交叉验证的方法来评估每种参数组合的性能
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         svm = SVC(gamma=gamma, C=C)
#         # 执行交叉验证
#         score = cross_val_score(svm, X_trainval, y_trainval, cv=5)
#         # 计算交叉验证平均精度
#         score = np.mean(score)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C': C, 'gamma': gamma}
#     # 在训练集+验证集上重新构建一个模型
#     svm = SVC(**best_parameters)
#     svm.fit(X_trainval, y_trainval)

# 下面的可视化说明了上述代码如何选择最佳参数设置
mglearn.plots.plot_cross_val_selection()
plt.show()

# 划分数据、运行网格搜索并评估最终参数的整体过程如图
mglearn.plots.plot_grid_search_overview()
plt.show()

# 由于带交叉验证的网格搜索时一种常用的调参方法，sklearn提供了GridSearch类
# 以估计器的形式实现了这种方法

# 字典的值是我们要调节的参数名称
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# 实例
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
# GridSearchCV将使用交叉验证来代替之前用过的数据集、测试集的方法
# 我们依然需要将数据划分为数据集和测试集，以免参数过拟合
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
# 我们创建的grid_search对象的行为像是一个分类器
grid_search.fit(X_train, y_train)
# GridSearchCV对象会搜索最佳参数，还会利用最佳交叉验证性能的参数在整个训练集上自动拟合一个新模型
# 为了评估找到的最佳参数的泛化能力，我们可以在测试集上调用score
print('Test set score:{:.2f}'.format(grid_search.score(X_test, y_test)))
# Test set score:0.97

# 在这里我们没有使用测试集来选择参数，也可以查看最佳参数和最佳精度
print('Best parameters:{}'.format(grid_search.best_params_))
print('Best cross-validation score:{:.2f}'.format(grid_search.best_score_))
# Best parameters:{'C': 10, 'gamma': 0.1}
# Best cross-validation score:0.97
# 注意这里调用score方法得到的分数是在整个训练集上训练到的模型
# best_score_属性保存的是交叉验证的平均精度，是在训练集上进行交叉验证得到的

# 访问实际找到的模型
print('Best estimator:\n{}'.format(grid_search.best_estimator_))
# Best estimator:
# SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

# 分析交叉验证的结果
# 网格搜索的结果可以在cv_results_属性中找到
result = pd.DataFrame(grid_search.cv_results_)
# 显示前5行
print(result.head())
#    mean_fit_time  std_fit_time  ...  std_test_score  rank_test_score
# 0       0.003940      0.004086  ...        0.022485               22
# 1       0.002108      0.001476  ...        0.022485               22
# 2       0.002027      0.000650  ...        0.022485               22
# 3       0.001450      0.001039  ...        0.022485               22
# 4       0.001473      0.000804  ...        0.022485               22
#
# [5 rows x 15 columns]
# 每一行对应一种参数设置，交叉验证所有的划分结果都被记录下来
# 将其可视化，首先提取平均验证分数，然后改变形状，使其坐标轴分别对应于C和gamma
score = np.array(result.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(score, xlabel='gamma', xticklabels=param_grid['gamma'],
                      ylabel='C', yticklabels=param_grid['C'], cmap='viridis')
plt.show()


# 下面通过热力图显示错误的搜索网格（参数范围问题）
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
# 创建不同的参数字典
param_grid_linear = {'C': np.linspace(1, 2, 6),
                     'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6),
                      'gamma': np.logspace(-3, 2, 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6),
                    'gamma': np.logspace(-7, -2, 6)}
for param_grid, ax, in zip([param_grid_linear, param_grid_one_log,
                            param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

    # 对交叉验证平均分数作图
    score_image = mglearn.tools.heatmap(
        scores, xlabel='gamma', ylabel='C', xticklabels=param_grid['gamma'],
        yticklabels=param_grid['C'], cmap='viridis', ax=ax
    )
plt.colorbar(score_image, ax=axes.tolist())
plt.show()
# 第一张图没什么变化，可以扩展更大更极端的范围

# 在非网格的空间搜索
# 在某些情况下需要调节的参数并不相同
# 核SVM分类器会根据所选核的不同而需要调节不同的参数，此时网格空间搜索不太实用
# para_grid可以是由字典组成的列表
param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
print('List of grids:\n{}'.format(param_grid))
# List of grids:
# [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100],
# 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}, {'kernel': ['linear'],
# 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

# 应用这个更加复杂的参数搜索
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best parameters:{}'.format(grid_search.best_params_))
print('Best cross-validation score:{}'.format(grid_search.best_score_))
# Best parameters:{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
# Best cross-validation score:0.9731225296442687

# 使用不同的交叉验证策略进行网格搜索
# GridSearchCV对分类问题默认使用分层k折交叉验证，对回归问题使用k折交叉验证
# （1）嵌套交叉验证
# 在嵌套交叉验证中，有一个外层循环，遍历将数据划分为训练集和测试集的所有划分
# 对每种划分都进行一次网格搜索，然后对每种外层划分，利用最佳参数设置计算测试集分数
# 嵌套验证不提供可用于新数据的模型
# 常用于评估给定模型在特定数据集上的效果
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                         iris.data, iris.target, cv=5)
print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())
# Cross-validation scores: [0.96666667 1.         0.9        0.96666667 1.        ]
# Mean cross-validation score: 0.9666666666666668


# 不好理解，简化理解代码如下
# def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
#     out_scores = []
#     # 对于外层交叉验证的每次数据划分，split方法返回索引值
#     for training_samples, test_samples in outer_cv.split(X, y):
#         # 利用内层交叉验证找到最佳参数
#         best_parms = {}
#         best_score = -np.inf
#         # 遍历参数
#         for parameters in parameter_grid:
#             # 在内层划分中累加分数
#             cv_scores = []
#             # 遍历内层交叉验证
#             for inner_train, inner_test in inner_cv.split(
#                 X[training_samples], y[training_samples]
#             ):
#                 # 对于给定的参数和训练数据来构建分类器
#                 clf = Classifier(**parameters)
#                 clf.fit(X[inner_train], y[inner_train])
#                 # 在内层测试集上进行评估
#                 score = clf.score(X[inner_test], y[inner_test])
#                 cv_scores.append(score)
#             # 计算交叉验证的平均分数
#             mean_score = np.mean(cv_scores)
#             if mean_score > best_score:
#                 best_score = mean_score
#                 best_parms = parameters
#         # 利用外层训练集和最佳参数来构建模型
#         clf = Classifier(**best_parms)
#         clf.fit(X[training_samples], y[training_samples])
#         # 评估模型
#         outer_score.append(clf.score(X[test_samples],
#                                      y[test_samples]))
#
#     return np.array(outer_score)
#
#
# # 在iris数据集上运行这个函数
# scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),
#                    StratifiedKFold(5), SVC, ParameterGrid(param_grid))
# print('Cross-validation scores:{}'.format(scores))

# (2)交叉验证与网格搜索并行

# 评估指标与评分
# 二分类指标
# 假正例、假反例
# 不平衡数据集是常态
# 为了便于说明，我们将digits数据集中的数字9与其他九个类别加一区分，创建一个9:1的不平衡数据集
digits = load_digits()
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0
)
# 我们可以使用DummyClassifier来始终预测多数项（非9）
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print('Unique predicted labels:{}'.format(np.unique(pred_most_frequent)))
print('Test score:{:.2f}'.format(dummy_majority.score(X_test, y_test)))
# Unique predicted labels:[False]
# Test score:0.90
# 我们得到了接近90％的精度，却没学到任何内容

# 我们将这个结果与一个真实分类器结果对比
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print('Test score:{:.2f}'.format(tree.score(X_test, y_test)))
# Test score:0.92

# 变化不大，我们再评估两个分类器

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print('dummy score:{:.2f}'.format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print('logre score:{:.2f}'.format(logreg.score(X_test, y_test)))
# dummy score:0.84
# logre score:0.98

# 可以看到，即使是最差的模型，精度都超过了80％
# 要相对不平衡数据集预测性能进行量化，精度并不是一个合适的度量

# 混淆矩阵
# 对于二分类问题的评估结果，一种最全面的方法是使用混淆矩阵
confusion = confusion_matrix(y_test, pred_logreg)
print('Confusion matrix:\n{}'.format(confusion))
# Confusion matrix:
# [[402   1]
#  [  6  41]]
# 注意理解对混淆矩阵的含义
mglearn.plots.plot_binary_confusion_matrix()
plt.show()

# 用混淆矩阵比较前面的结果
print('Most frequent class:')
print(confusion_matrix(y_test, pred_most_frequent))
print('\nDummy model:')
print(confusion_matrix(y_test, pred_dummy))
print('\nDecision tree:')
print(confusion_matrix(y_test, pred_tree))
print('\nLogistic Regression')
print(confusion_matrix(y_test, pred_logreg))
# Most frequent class:
# [[403   0]
#  [ 47   0]]
#
# Dummy model:
# [[361  42]
#  [ 43   4]]
#
# Decision tree:
# [[390  13]
#  [ 24  23]]
#
# Logistic Regression
# [[402   1]
#  [  6  41]]


# 其他评估指标还有准确率、召回率与f分数
# 准确率度量的是预测为正类的样本中有多少真正的正例
# precision = TP/(TP + FP)
# 如果目标是限制假正例的数量，可以使用准确率作为指标
# 准确性也被称为阳性预测值（PPV）

# 召回率度量的是正类样本中有多少被预测为正类
# Recall = TP/(TP + FN)

# 召回率和准确率是最常用的二分类评估器

# 只看召回率与准确率无法提供完整的图景
# 将两者进行汇总的方法是f-分数或f-度量
# 是准确性与召回率的调和平均
print('f1 score most frequent:{:.2f}'.format(
    f1_score(y_test, pred_most_frequent)
))
print('f1 score dummy:{:.2f}'.format(f1_score(y_test, pred_dummy)))
print('f1 score tree:{:.2f}'.format(f1_score(y_test, pred_tree)))
print('f1 score logistic regression: {:.2f}'.format(
    f1_score(y_test, pred_logreg)
))
# f1 score most frequent:0.00
# f1 score dummy:0.09
# f1 score tree:0.55
# f1 score logistic regression: 0.92

# 如果我们想要对准确率、召回率、f-分数做一个更全面的总结，可以使用classification_report这个方便的函数
print(classification_report(y_test, pred_most_frequent,
                            target_names=['not nine', 'nine']))
# classification_report为每个类别（True, False）生成一行，并给出以该类作为正类的准确率、召回率、f-分数
#               precision    recall  f1-score   support
#
#     not nine       0.90      1.00      0.94       403
#         nine       0.00      0.00      0.00        47
#
#     accuracy                           0.90       450
#    macro avg       0.45      0.50      0.47       450
# weighted avg       0.80      0.90      0.85       450

# 考虑不确定性
# 混淆矩阵和分类报告都提供了非常详细的分析
# 但预测本身已经丢弃了模型中包含的大量信息
# 下面一个不平衡二分类任务中，反类有400个点，正类只有50个点
# 我们在数据上训练一个核SVM模型
# 你可以在图像偏上的位置看到一个黑色圆圈，表示decision_function的阈值刚好为0
# 圆内的点划为正类
# X, y = make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2], random_state=22)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# svc = SVC(gamma=.05).fit(X_train, y_train)
mglearn.plots.plot_decision_threshold()
plt.show()

# 以下代码详见教材
# 准确率-召回率曲线
# precision, recall, thresholds = precision_recall_curve(
#     y_test, svm.decision_function(X_test)
# )
# 返回一个列表，包含按顺序排列的所有可能阈值对应的准确率和召回率
# 可以由此绘制曲线

# 受试者工作特征曲线（ROC曲线）与AUC
# ROC曲线考虑了给定分类器的所有可能的阈值，显示了假正例率和真正例率，而不是报告准确率和召回率
# 对于不平衡的分类问题来说，AUC是一个比精度好得多的指标
# 回到前面的例子
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0
)
plt.figure()
for gamma in [1, 0.05, 0.01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
    print('gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}'.format(
        gamma, accuracy, auc
    ))
    plt.plot(fpr, tpr, label='gamma = {:.3f}'.format(gamma))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc='best')
plt.show()
# gamma = 1.00  accuracy = 0.90  AUC = 0.50
# gamma = 0.05  accuracy = 0.90  AUC = 1.00
# gamma = 0.01  accuracy = 0.90  AUC = 1.00
# AUC等于1说明可以通过调节阈值来得到完美的结果（这里阈值需要自己调节）

# 多分类指标
# 混淆矩阵
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=0
)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print('Accuracy:{:.3f}'.format(accuracy_score(y_test, pred)))
print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, pred)))
# Accuracy:0.951
# Confusion matrix:
# [[37  0  0  0  0  0  0  0  0  0]
#  [ 0 40  0  0  0  0  0  0  2  1]
#  [ 0  1 40  3  0  0  0  0  0  0]
#  [ 0  0  0 43  0  0  0  0  1  1]
#  [ 0  0  0  0 37  0  0  1  0  0]
#  [ 0  0  0  0  0 46  0  0  0  2]
#  [ 0  1  0  0  0  0 51  0  0  0]
#  [ 0  0  0  1  1  0  0 46  0  0]
#  [ 0  3  1  0  0  0  0  0 43  1]
#  [ 0  0  0  0  0  1  0  0  1 45]]
# 每一行对应真实标签
# 每一列对应预测标签

# 分类报告
print(classification_report(y_test, pred))
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        37
#            1       0.89      0.93      0.91        43
#            2       0.98      0.91      0.94        44
#            3       0.91      0.96      0.93        45
#            4       0.97      0.97      0.97        38
#            5       0.98      0.96      0.97        48
#            6       1.00      0.98      0.99        52
#            7       0.98      0.96      0.97        48
#            8       0.91      0.90      0.91        48
#            9       0.90      0.96      0.93        47
#
#     accuracy                           0.95       450
#    macro avg       0.95      0.95      0.95       450
# weighted avg       0.95      0.95      0.95       450

# f-分数，对于多分类的f分数，对每个类别计算一个二分类f-分数，该类视为正类
# 然后使用以下方法做平均
# 宏平均（直接返回每个二分类f分数作同等权重平均）
# 加权平均，分类报告给出这个值
# 微平均
# 如果对样本同等对待，推荐使用微平均
# 如果对类别同等对待，推荐使用宏平均
print('Micro average f1 score:{:.3f}'.format(
    f1_score(y_test, pred, average='micro')
))
print('Macro average f1 score:{:.3f}'.format(
    f1_score(y_test, pred, average='macro')
))
# Micro average f1 score:0.951
# Macro average f1 score:0.952

# 回归指标
# 对于一般回归问题，R2指标足够矣
# 在模型选择中使用评估指标
# 我们希望使用GridSearchCV和cross_cal_score模型时使用AUC等指标
# 以上要求可以由scoring参数实现
# 实例

# 分类问题的默认评分是精度
print('Default scoring:{}'.format(
    cross_val_score(SVC(), digits.data, digits.target == 9)
))
# 指定'scoring = 'accuracy''不会改变结果
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9,
                                    scoring='accuracy')
print('Explicit accuracy scoring:{}'.format(explicit_accuracy))
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9,
                          scoring='roc_auc')
print('AUC scoring:{}'.format(roc_auc))
# Default scoring:[0.975      0.99166667 1.         0.99442897 0.98050139]
# Explicit accuracy scoring:[0.975      0.99166667 1.         0.99442897 0.98050139]
# AUC scoring:[0.99717078 0.99854252 1.         0.999828   0.98400413]

# 类似地， 可以改变GridSearchCV中用于选择最佳参数的指标
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0
)
# 我们给出了不太好的网格来说明
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
# 使用默认的参数
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print('Grid-Search with accuracy')
print('Best parameter:', grid.best_params_)
print('Best cross-validation score(accuracy):{:.3f}'.format(grid.best_score_))
print('Test set AUC:{:.3f}'.format(
    roc_auc_score(y_test, grid.decision_function(X_test))
))
print('Test set accuracy:{:.3f}'.format(grid.score(X_test, y_test)))
# Grid-Search with accuracy
# Best parameter: {'gamma': 0.0001}
# Best cross-validation score(accuracy):0.976
# Test set AUC:0.992
# Test set accuracy:0.973

# 使用AUC评分来代替
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring='roc_auc')
grid.fit(X_train, y_train)
print('\nGrid-Search with AUC')
print('Best parameters:', grid.best_params_)
print('Best cross-validation score(AUC):{:.3f}'.format(grid.best_score_))
print('Test set AUC:{:.3f}'.format(
    roc_auc_score(y_test, grid.decision_function(X_test))
))
print('Test set accuracy:{:.3f}'.format(grid.score(X_test, y_test)))
# Grid-Search with AUC
# Best parameters: {'gamma': 0.01}
# Best cross-validation score(AUC):0.998
# Test set AUC:1.000
# Test set accuracy:1.000

# 查看所有scoring取值
print('Available scores:\n{}'.format(sorted(SCORERS.keys())))

