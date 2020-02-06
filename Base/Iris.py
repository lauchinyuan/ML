from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


# 获取自带的数据集
iris_data = load_iris()  # 返回Bunch对象，类似字典
print('key for iris_data:\n{}'.format(iris_data.keys()))
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
# 查看数据集说明
print(iris_data['DESCR'])
# 查看花的种类
print('Target name:{}'.format(iris_data['target_names']))
# Target name:['setosa' 'versicolor' 'virginica']
# 获取特征说明
print(iris_data['feature_names'])
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print('Shape of data:{}'.format(iris_data['data'].shape))  # 查看数据集大小
# Shape of data:(150, 4)
print('First five rows of data:\n{}'.format(iris_data['data'][:5]))  # 查看部分数据
# First five rows of data:
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

# 分离训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)

# 将类实例化为一个对象,knn对象对算法进行了封装
knn = KNeighborsClassifier(n_neighbors=1)  # 设定邻居数为1
# 基于训练集构建模型
knn.fit(X_train, y_train)

# 作出预测
X_new = np.array([[5, 2.9, 3.0, 0.2]])  # 假设的新样本数据,传入的需为2维数组
prediction = knn.predict(X_new)
print('Prediction:{}'.format(prediction))
# Prediction:[0]
print('Predict name:{}'.format(iris_data['target_names'][prediction]))
# Predict name:['versicolor']

# 评估模型
# 利用模型对测试集进行评估
y_pred = knn.predict(X_test)
print('Test set score:{}'.format(np.mean(y_test == y_pred)))
# Test set score:0.9736842105263158
print('Test set score:{}'.format(knn.score(X_test, y_test)))
# Test set score:0.9736842105263158



