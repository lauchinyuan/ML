import matplotlib
import matplotlib.pyplot as plt
import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# 举例子说明模型链的重要性
# 已知可以通过MinMaxScaler来进行预处理来提高SVM的性能
# 代码如下
# 加载并划分数据
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

# 计算训练数据的最小值和最大值
scaler = MinMaxScaler().fit(X_train)

# 对训练数据进行缩放
X_train_scaled = scaler.transform(X_train)
svm = SVC()
svm.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
print('Test score:{:.2f}'.format(svm.score(X_test_scaled, y_test)))
# Test score:0.97

# 用预处理进行参数选择
# 希望利用GridSearchCV招到更好的SVC参数
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print('Best cross-validation accuracy:{:.2f}'.format(grid.best_score_))
print('Best set score:{:.2f}'.format(grid.score(X_test_scaled, y_test)))
print('Best parameter:', grid.best_params_)
# Best cross-validation accuracy:0.98
# Best set score:0.97
# Best parameter: {'C': 1, 'gamma': 1}

# 应在进行任何预处理之前完成数据集的划分
# pipeline可以讲多个处理步骤合并为单个sklearn估计器
# pipeline将预处理步骤与一个监督模型链接在一起

# 构建管道
# 构建一个由步骤列表组成的管道对象，每个对象都是一个元组
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
pipe.fit(X_train, y_train)
# 这里首先对第一个步骤（缩放器）调用fit, 然后使用该缩放器对训练数据进行缩放，最后用缩放后的数据拟合SVM
print('Test score:{:.2f}'.format(pipe.score(X_test, y_test)))
# Test score:0.97

# 在网格搜索中使用管道
# 为管道定义参数网格的语法是为每个参数定义一个步骤名称，后面加上__
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# 接下来可以使用GridSearchCV
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print('Best cross-validation accuracy:{:.2f}'.format(grid.best_score_))
print('Test set score:{:.2f}'.format(grid.score(X_test, y_test)))
print('Best parameters:{}'.format(grid.best_params_))
# Best cross-validation accuracy:0.98
# Test set score:0.97
# Best parameters:{'svm__C': 1, 'svm__gamma': 1}
# 与前面的网格搜索不同，现在对于交叉验证的每次划分来说，仅使用训练部分对MinMaxScaler进行拟合

# 通用的管道接口
# 对于管道内的估计器的唯一要求就是除了最后一步之外的所有步骤都具有Transform方法
# 注意pipeline.step是由元组组成的列表
# pipeline.step[0][1]是第一个估计器，pipeline.step[1][1]是第二个估计器

# 用make_pipeline方便地创建管道
# 标准语法
pipe_long = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC(C=100))])
# 缩写语法
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
# pipe_short自动命名
# 查看步骤名称
print('Pipeline steps:\n{}'.format(pipe_short.steps))
# [('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False))]

# 通常的命名时类名称的小写版本
# 如果多个步骤属于同一类，则会附加一个数字
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print('Pipeline steps:\n{}'.format(pipe.steps))
# Pipeline steps:
# [('standardscaler-1', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#     svd_solver='auto', tol=0.0, whiten=False)), ('standardscaler-2', StandardScaler(copy=True, with_mean=True, with_std=True))]

# 访问步骤属性
# 用前面定义的管道对cancer数据集进行拟合
pipe.fit(cancer.data)
# 从'pca'步骤中提取前两个主成分
components = pipe.named_steps['pca'].components_
print('components.shape:{}'.format(components.shape))
# components.shape:(2, 30)

# 访问网格搜索管道中的属性
pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=4
)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print('Best estimation:\n{}'.format(grid.best_estimator_))
# Best estimation:
# Pipeline(memory=None,
#          steps=[('standardscaler',
#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#                 ('logisticregression',
#                  LogisticRegression(C=1, class_weight=None, dual=False,
#                                     fit_intercept=True, intercept_scaling=1,
#                                     l1_ratio=None, max_iter=100,
#                                     multi_class='auto', n_jobs=None,
#                                     penalty='l2', random_state=None,
#                                     solver='lbfgs', tol=0.0001, verbose=0,
#                                     warm_start=False))],
#          verbose=False)


print('Logistic regression step:\n{}'.format(
    grid.best_estimator_.named_steps['logisticregression']
))
# Logistic regression step:
# LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
#                    warm_start=False)

# 网格搜索预处理步骤与模型参数
# 下面用一个管道实现三个步骤：数据缩放、计算多项式、岭回归
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
# 我们希望根据分类结果来选择degree参数，我们可以利用管道搜索degree参数及Ridge的alpha参数
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
# 现在可以运行网格搜索
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# 我们可以用热图将交叉验证的结果可视化
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),
            vmin=0, cmap='viridis')
plt.xlabel('ridge__alpha')
plt.ylabel('polynomialfeatures__degree')
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])),
           param_grid['polynomialfeatures__degree'])
plt.colorbar()
plt.show()

# 查看最佳参数
print('Best parameters:{}'.format(grid.best_params_))
print('Test-set score:{:.2f}'.format(grid.score(X_test, y_test)))
# Best parameters:{'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
# Test-set score:0.77
# 为了对比，运行一个没有多项式特征的网格搜索
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print('Score without poly features:{:.2f}'.format(grid.score(X_test, y_test)))
# Score without poly features:0.63

# 网格搜索选择使用哪个模型
# 对于不同的模型，所需的步骤通常有所不同
# 定义一个管道
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
param_grid = [{'classifier': [SVC()],
               'preprocessing': [StandardScaler(), None],
               'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
               'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'classifier': [RandomForestClassifier(n_estimators=100)],
               'preprocessing': [None],
               'classifier__max_features': [1, 2, 3]}]
# 现在可以像前面一样将网格搜索实例化并在数据集上运行
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print('Best params:\n{}\n'.format(grid.best_params_))
print('Best cross-validation score:{:.2f}'.format(grid.best_score_))
print('Test-set score:{:.2f}'.format(grid.score(X_test, y_test)))
# Best params:
# {'classifier': SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False), 'classifier__C': 10, 'classifier__gamma': 0.01, 'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}
#
# Best cross-validation score:0.99
# Test-set score:0.98

