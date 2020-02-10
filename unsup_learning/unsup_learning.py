import matplotlib
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
import numpy as np
# 预处理与缩放
mglearn.plots.plot_scaling()
plt.show()
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)
scaler = MinMaxScaler()  # 实例化
# 使用fit方法拟合缩放器
scaler.fit(X_train)

# 变换数据
X_train_scaled = scaler.transform(X_train)
# 在缩放之前和之后分别打印数据集
print('transformed shape:{}'.format(X_train_scaled.shape))
print('per-feature minimum before scaling:\n{}'.format(X_train.min(axis=0)))
print('per-feature maximum before scaling:\n{}'.format(X_train.max(axis=0)))
print('per-feature minimum after scaling:\n{}'.format(X_train_scaled.min(axis=0)))
print('per-feature maximum after scaling:\n{}'.format(X_train_scaled.max(axis=0)))
# transformed shape:(426, 30)
# per-feature minimum before scaling:
# [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
#  0.000e+00 1.060e-01 4.996e-02 1.115e-01 3.628e-01 7.570e-01 7.228e+00
#  1.713e-03 2.252e-03 0.000e+00 0.000e+00 7.882e-03 8.948e-04 7.930e+00
#  1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
#  1.565e-01 5.504e-02]
# per-feature maximum before scaling:
# [2.811e+01 3.381e+01 1.885e+02 2.501e+03 1.447e-01 3.114e-01 4.268e-01
#  2.012e-01 3.040e-01 9.744e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
#  2.333e-02 1.064e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
#  4.954e+01 2.512e+02 4.254e+03 2.226e-01 1.058e+00 1.252e+00 2.903e-01
#  6.638e-01 2.075e-01]
# per-feature minimum after scaling:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0.]
# per-feature maximum after scaling:
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1.]

# 对测试集进行变换
X_test_scaled = scaler.transform(X_test)
# 打印缩放后数据集的属性
print('per-feature minimum after scaling:\n{}'.format(X_test_scaled.min(axis=0)))
print('per-feature maximum after scaling:\n{}'.format(X_test_scaled.max(axis=0)))
# per-feature minimum after scaling:
# [ 0.03540158  0.04190871  0.02895446  0.01497349  0.14260888  0.04999658
#   0.          0.          0.07222222  0.00589722  0.00105015 -0.00057494
#   0.00067851 -0.0007963   0.05148726  0.01434497  0.          0.
#   0.04195752  0.01113138  0.03678406  0.01252665  0.03366702  0.01400904
#   0.08531995  0.01833687  0.          0.          0.00749064  0.02367834]
# per-feature maximum after scaling:
# [0.76809125 1.22697095 0.75813696 0.64750795 1.20310633 1.11643038
#  0.99906279 0.90606362 0.93232323 0.94903117 0.45573058 0.72623944
#  0.48593507 0.31641282 1.36082713 1.2784499  0.36313131 0.77476795
#  1.32643996 0.72672498 0.82106012 0.87553305 0.77887345 0.67803775
#  0.78603975 0.87843331 0.93450479 1.0024113  0.76384782 0.58743277]

# 对训练集数据和测试集数据作相同的缩放
# 构造数据
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 数据划分
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# 绘制训练集和测试集
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=mglearn.cm2(1), label='Training set', s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=mglearn.cm2(1), label='Test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('Original Data')

# 缩放数据
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将正确所放的数据可视化
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2(0), label='Training set', s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
                c=mglearn.cm2(0), label='Test set', s=60)
axes[1].set_title('Scaled Data')

for ax in axes:
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
plt.show()

# 预处理对监督学习的作用
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print('Test set accuracy:{:.3f}'.format(svm.score(X_test, y_test)))
# Test set accuracy:0.944

# 使用预处理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 在缩放后的数据集上学习SVM
svm.fit(X_train_scaled, y_train)
print('Scaled test set accuracy:{:.2f}'.format(
    svm.score(X_test_scaled, y_test)
))
# Scaled test set accuracy:0.97

# 其他预处理方法
# 利用零均值和单位方差的缩放方法进行预处理
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print('SVM test accuracy:{:.2f}'.format(svm.score(X_test_scaled, y_test)))
# SVM test accuracy:0.96

# 降维、特征提取与流形学习
# 主成分分析（PCA）
mglearn.plots.plot_pca_illustration()
plt.show()

# 将PCA应用于cancer数据集并可视化
# PCA最常见的应用之一就是将高维数据可视化

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer['data'][cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()
for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel('Feature magnitude')
ax[0].set_ylabel('Frequency')
ax[0].legend(['malignant', 'benign'], loc='best')
fig.tight_layout()
plt.show()

# 直接使用二维空间可视化
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
# 指定想要保留的主成分的个数
pca = PCA(n_components=2)
pca.fit(X_scaled)
# 将数据变换到前两个主成分的方向上
X_pca = pca.transform(X_scaled)
print('Original shape:{}'.format(X_scaled.shape))
print('Reduced shape:{}'.format(X_pca.shape))
# Original shape:(569, 30)
# Reduced shape:(569, 2)

# 对前两个主成分作图
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

# 主成分实质上是所有特征的混合
# 查看PAC对象的主成分
print('PCA components:\n{}'.format(pca.components_))
# PCA components:
# [[ 0.21890244  0.10372458  0.22753729  0.22099499  0.14258969  0.23928535
#    0.25840048  0.26085376  0.13816696  0.06436335  0.20597878  0.01742803
#    0.21132592  0.20286964  0.01453145  0.17039345  0.15358979  0.1834174
#    0.04249842  0.10256832  0.22799663  0.10446933  0.23663968  0.22487053
#    0.12795256  0.21009588  0.22876753  0.25088597  0.12290456  0.13178394]
#  [-0.23385713 -0.05970609 -0.21518136 -0.23107671  0.18611302  0.15189161
#    0.06016536 -0.0347675   0.19034877  0.36657547 -0.10555215  0.08997968
#   -0.08945723 -0.15229263  0.20443045  0.2327159   0.19720728  0.13032156
#    0.183848    0.28009203 -0.21986638 -0.0454673  -0.19987843 -0.21935186
#    0.17230435  0.14359317  0.09796411 -0.00825724  0.14188335  0.27533947]]

# 将主成分系数可视化
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['First component', 'Second component'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel('Feature')
plt.ylabel('Principal component')
plt.show()

# 特征提取的特征脸
# PCA另一个应用便是特征提取
# 特征提取背后的思想:可以找到一种数据表示，比原始解释更适合于分析
# 获取数据集数据
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()}
                         )
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()

print('people.images.shape:{}'.format(people.images.shape))
print('Number of classes:{}'.format(len(people.target_names)))
# people.images.shape:(3023, 87, 65)
# Number of classes:62

# 计算每个目标出现的次数
counts = np.bincount(people.target)
# 将次数与目标名一起打印出来
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print('{0:25} {1:3}'.format(name, count), end='    ')
    if(i + 1) % 3 == 0:
        print()

# Alejandro Toledo           39    Alvaro Uribe               35    Amelie Mauresmo            21
# Andre Agassi               36    Angelina Jolie             20    Ariel Sharon               77
# Arnold Schwarzenegger      42    Atal Bihari Vajpayee       24    Bill Clinton               29
# Carlos Menem               21    Colin Powell              236    David Beckham              31
# Donald Rumsfeld           121    George Robertson           22    George W Bush             530
# Gerhard Schroeder         109    Gloria Macapagal Arroyo    44    Gray Davis                 26
# Guillermo Coria            30    Hamid Karzai               22    Hans Blix                  39
# Hugo Chavez                71    Igor Ivanov                20    Jack Straw                 28
# Jacques Chirac             52    Jean Chretien              55    Jennifer Aniston           21
# Jennifer Capriati          42    Jennifer Lopez             21    Jeremy Greenstock          24
# Jiang Zemin                20    John Ashcroft              53    John Negroponte            31
# Jose Maria Aznar           23    Juan Carlos Ferrero        28    Junichiro Koizumi          60
# Kofi Annan                 32    Laura Bush                 41    Lindsay Davenport          22
# Lleyton Hewitt             41    Luiz Inacio Lula da Silva  48    Mahmoud Abbas              29
# Megawati Sukarnoputri      33    Michael Bloomberg          20    Naomi Watts                22
# Nestor Kirchner            37    Paul Bremer                20    Pete Sampras               22
# Recep Tayyip Erdogan       30    Ricardo Lagos              27    Roh Moo-hyun               32
# Rudolph Giuliani           26    Saddam Hussein             23    Serena Williams            52
# Silvio Berlusconi          33    Tiger Woods                23    Tom Daschle                25
# Tom Ridge                  33    Tony Blair                144    Vicente Fox                32
# Vladimir Putin             49    Winona Ryder               24

# 为了降低数据偏斜，对每个人最多取50张图像
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

# 将灰度值缩放到0到1之间，而不是0到255之间
# 以得道更好的数据稳定性
X_people = X_people / 255

# K近邻用于人脸识别
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people,
                                                    stratify=y_people,
                                                    random_state=0)
# 使用一个邻居
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print('Test set score of 1-nn:{:.2f}'.format(knn.score(X_test, y_test)))
# Test set score of 1-nn:0.23
# 效果较差

# PAC白化，将主成分相同的尺度
mglearn.plots.plot_pca_whitening()
plt.show()

# 对训练数据拟合PCA对象，提取前100个主成分，然后对训练数据和测试数据进行变换
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('X_train.shape:{}'.format(X_train.shape))
print('X_train_pca.shape:{}'.format(X_train_pca.shape))
# X_train.shape:(1547, 5655)
# X_train_pca.shape:(1547, 100)
# 新数据有100个特征，现在可以对新表示使用单一最近邻分类器来对图像分类
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print('Test set accuracy:{:.2f}'.format(knn.score(X_test_pca, y_test)))
# 成分提供了更好的数据表示，精度有显著提高
# Test set accuracy:0.31

# 查看前几个主成分
print('pca.component_.shape:{}'.format(pca.components_.shape))
# pca.component_.shape:(100, 5655)
fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title('{}.component'.format((i + 1)))
plt.show()

# 非负矩阵分解（NMF）
# 与PCA相比，NMF得到的分量更容易解释
# 负的系数可能会导致难以解释的抵消效应
# NMF在二维数据上的结果
mglearn.plots.plot_nmf_illustration()
plt.show()
# NMF会删除一些方向，并且会创建一组完全不同的方向
# 将NMF应用于图像数据

# 尝试提取一部分分量，初步观察数据
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fix, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title('{}.component'.format(i))
plt.show()

# 可以看到分量三人脸向右转，分量七人脸向左转
# 查看这两个分量特别大的图像
compn = 3
# 按照第三个分量排序，绘制前10张图片
inds = np.argsort(X_train_nmf[:, compn])[::-1]  # [::-1]表示倒序
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()

# 查看原始信号源
S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

# 将数据混合成100维的状态
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print('Shape of measurements:{}'.format(X.shape))
#

# 可以用NMF来还原这三个信号
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print('Recovered signal shape:{}'.format(S_.shape))
#

# 使用PCA作为对比
pca = PCA(n_components=3)
H = pca.fit_transform(X)

# 将信号可视化
models = [X, S, S_, H]
names = ['Observations(first three measurements)',
         'True source',
         'NMF recovered signals',
         'PCA recovered signals']
fig, axes = plt.subplots(4, figsize=(8, 4))
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
plt.show()

# 用t-SNE进行流形学习
# 流形学习对于探索性数据分析很有用
# t-SNE重点关注距离较近的点
digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
plt.show()

# 用PCA将降到二维数据可视化
# 构建一个PCA模型
pca = PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525',
          '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    # 将数据实际绘制文本，而不是散点
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

# 使用t-SNE使用于数据集
tsne = TSNE(random_state=42)
# 使用fit_tranform而不是fit， 因为TSNE没有tranform方法
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.show()

# 聚类
# K均值聚类
mglearn.plots.plot_kmeans_algorithm()
plt.show()

mglearn.plots.plot_kmeans_boundaries()
plt.show()

# 生成模拟的二维数据
X, y = make_blobs(random_state=1)
# 构建聚类模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print('Cluster memberships:\n{}'.format(kmeans.labels_))
# Cluster memberships:
# [0 1 1 1 2 2 2 1 0 0 1 1 2 0 2 2 2 0 1 1 2 1 2 0 1 2 2 0 0 2 0 0 2 0 1 2 1
#  1 1 2 2 1 0 1 1 2 0 0 0 0 1 2 2 2 0 2 1 1 0 0 1 2 2 1 1 2 0 2 0 1 1 1 2 0
#  0 1 2 2 0 1 0 1 1 2 0 0 0 0 1 0 2 0 0 1 1 2 2 0 2 0]

print(kmeans.predict(X))
# [0 1 1 1 2 2 2 1 0 0 1 1 2 0 2 2 2 0 1 1 2 1 2 0 1 2 2 0 0 2 0 0 2 0 1 2 1
#  1 1 2 2 1 0 1 1 2 0 0 0 0 1 2 2 2 0 2 1 1 0 0 1 2 2 1 1 2 0 2 0 1 1 1 2 0
#  0 1 2 2 0 1 0 1 1 2 0 0 0 0 1 0 2 0 0 1 1 2 2 0 2 0]

# 可视化
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
    markers='^', markeredgewidth=2
)
plt.show()
# 也可以使用更多或者更少的簇中心
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 使用两个簇中心
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignment = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment, ax=axes[0])

# 使用五个簇中心
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignment = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment, ax=axes[1])
plt.show()

# k均值的失败案例, k均值无法识别非球形簇
# 生成一些随机的分组数据
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
# 变换数据使其拉长
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)
# 将数据聚类成3个簇
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 画出簇分配和簇中心
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[0, 1, 2], s=100, linewidths=2, cmap=mglearn.cm3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 如果簇的形状更加复杂，k均值的表现也很差
# 生成模拟的数据（这次噪声较少）
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 将数据聚类成2个簇
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# 画出簇分配和簇中心
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidths=2)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 矢量量化，或者将K均值看做分解
# 并排比较PCA、NMF和k均值
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0
)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_people)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_rec_pca = pca.inverse_transform(pca.transform(X_test))
X_rec_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_rec_nmf = np.dot(nmf.transform(X_test), nmf.components_)

# fig, axes = plt.subplots(3, 5, figsize=(8, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for ax, comp_kmeans, comp_pca, comp_nmf in zip(
#     axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
#     ax[0].imshow(comp_kmeans.reshape(image_shape))
#     ax[1].imshow(comp_pca.reshape(image_shape))
#     ax[2].imshow(comp_nmf.reshape(image_shape))
# axes[0, 0].set_ylabel('kmeans')
# axes[1, 0].set_ylabel('pca')
# axes[2, 0].set_ylabel('nmf')
#
# fig, axes = plt.subplots(4, 5, figsize=(8, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
#     axes.T, X_test, X_rec_kmeans, X_rec_pca, X_rec_nmf):
#     ax[0].imshow(orig.reshape(image_shape))
#     ax[1].imshow(rec_kmeans.reshape(image_shape))
#     ax[2].imshow(rec_pca.reshape(image_shape))
#     ax[3].imshow(rec_nmf.reshape(image_shape))
# axes[0, 0].set_ylabel('original')
# axes[1, 0].set_ylabel('kmeans')
# axes[2, 0].set_ylabel('pca')
# axes[3, 0].set_ylabel('nmf')
# plt.show()
#
# # K均值算法可以用比收入维度更多的簇来对数据进行编码
# # 可以找到一种更具表现力的表示
# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
#
# kmeans = KMeans(n_clusters=10, random_state=0)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)
#
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             s=60, marker='^', c=range(kmeans.n_clusters),
#             linewidths=2, cmap='Paired')
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# print('Cluster memberships:\n{}'.format(y_pred))
# plt.show()
# Cluster memberships:
# [4 7 6 9 7 7 4 1 4 1 8 3 7 1 0 4 2 3 2 0 5 1 2 1 8 6 7 5 6 2 4 8 1 6 4 5 3
#  4 0 6 3 8 2 6 7 8 4 0 6 1 0 3 5 9 1 4 2 1 2 8 3 9 7 4 1 9 8 7 8 9 3 9 3 6
#  1 9 6 4 2 3 5 8 3 5 6 8 4 8 3 5 2 4 5 0 5 7 7 3 9 6 1 5 8 4 9 6 9 8 7 2 0
#  8 8 9 4 1 2 5 3 4 4 0 6 8 6 0 4 6 1 5 4 0 9 3 1 7 1 9 5 4 6 6 2 8 8 4 6 1
#  2 6 3 7 4 2 3 8 1 3 2 2 6 1 2 7 3 7 2 3 7 1 2 9 0 0 6 1 5 0 0 2 7 0 5 7 5
#  2 8 3 9 0 9 2 4 4 6 0 5 6 2 7]

# k均值算法的缺点在于，它依赖于随机初始化
# 对簇形状的假设的约束性较强
# 要求指定所要寻找的簇的个数

# 凝聚聚类
# sklearn实现了以下三种选项
# ward挑选两个簇来合并，使所有簇的方差增加最小
# average将所有点之间平均距离最小的两个簇最小
# complete（最大链接）将簇中点之间最大距离最小的两个簇合并
# 查看构造过程
mglearn.plots.plot_agglomerative_algorithm()
plt.show()

X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 层次聚类与树状图
mglearn.plots.plot_agglomerative()
plt.show()

# DBSCAN
# DBSCAN不需要用户先验地设置簇的个数， 可以划分具有复杂形状的簇
# DBSCAN思想：簇形成数据的密集区域，并由相对较空的区域分隔开
# 在密集区域内的点称为核心样本
# 在给定数据点eps距离内至少有min_sample个点，则为核心样本
# DBSCAN将核心样本放置于同一个簇中
X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print('Cluster memberships:\n{}'.format(clusters))
# Cluster memberships:
# [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

mglearn.plots.plot_dbscan()
plt.show()

# DBSCAN在双月数据上运行
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 将数据缩放成平均值为0， 方差为1
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# 绘制簇分配
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()

# 聚类算法的对比与评估
# 用真实值评估聚类
# 评估指标：ARI NMI
# 用ARI比较k均值、凝聚算法、DBSCAN
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})
# 列出要使用的算法
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]
# 创建一个随机的簇分配，作为参考
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# 绘制随机分配
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3,
                s=60)
axes[0].set_title('Random assignment - ARI:{:.2f}'.format(adjusted_rand_score(y,
                                                                              random_clusters)))
for ax, algorithm in zip(axes[1:], algorithms):
    # 绘制簇分配和簇中心
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
               cmap=mglearn.cm3, s=60)
    ax.set_title('{}-ARI:{:.2f}'.format(algorithm.__class__.__name__,
                                        adjusted_rand_score(y, clusters)))
plt.show()

# 注意评估聚类时，不要使用accuracy_score，因为标签并无意义
# 两种点标签对应于相同的聚类
clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]
# 精度为0，因为两者标签完全不同
print('ARI:{:.2f}'.format(adjusted_rand_score(clusters1, clusters2)))
#

# 没有真实值的情况下评估聚类
# 轮廓系数，在实践中效果不好
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})
# 创建一个随机的簇分配，作为参考
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# 绘制随机分配
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60)
axes[0].set_title('Random assignment:{:.2f}'.format(
    silhouette_score(X_scaled, random_clusters)
))
for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3)
    ax.set_title('{}:{:.2f}'.format(algorithm.__class__.__name__,
                                    silhouette_score(X_scaled, clusters)))
plt.show()

# 对于聚类评估，稍好的方法是基于鲁棒性的聚类指标
# 要想知道聚类是否对应我们感兴趣的内容，唯一的方法是对聚类进行人工分析

# 在人脸数据上比较算法
# 将数据用特征脸表示，包含100个成分PCA生成
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

# 用DBSCAN分析人脸数据
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print('Unique labels:{}'.format(np.unique(labels)))
# Unique labels:[-1]

# 尝试改变min_samples
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print('Unique labels:{}'.format(np.unique(labels)))
# Unique labels:[-1]

# 再增大eps
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print('Unique labels:{}'.format(np.unique(labels)))
# Unique labels:[-1  0]

# 计算所有簇中点数和噪声中点数
# bincount不予许负值，所以我们需要加1
# 结果中的第一个数字对应于噪声点
print('Number of points per cluster:{}'.format(np.bincount(labels + 1)))
# Number of points per cluster:[  31 2032]

# 查看所有噪声点
noise = X_people[labels == -1]
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
plt.show()

# 查看eps不同取值对应的结果
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print('\neps={}'.format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print('Cluster present:{}'.format(np.unique(labels)))
    print('Cluster sizes:{}'.format(np.bincount(labels + 1)))
# eps=1
# Cluster present:[-1]
# Cluster sizes:[2063]
#
# eps=3
# Cluster present:[-1]
# Cluster sizes:[2063]
#
# eps=5
# Cluster present:[-1]
# Cluster sizes:[2063]
#
# eps=7
# Cluster present:[-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
# Cluster sizes:[2003    4   14    7    4    3    3    4    4    3    3    5    3    3]
#
# eps=9
# Cluster present:[-1  0  1  2]
# Cluster sizes:[1306  751    3    3]
#
# eps=11
# Cluster present:[-1  0]
# Cluster sizes:[ 413 1650]
#
# eps=13
# Cluster present:[-1  0]
# Cluster sizes:[ 120 1943]

# eps = 7结果最有趣， 有许多较小的簇，将13个较小的簇中的点全部可视化来深入研究这一聚类
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for image, label, ax in zip(X_people[mask], y_people[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])
plt.show()

# 用k均值分析人脸数据集
# 凝聚和k均值更可能创建更加均匀大小的簇
# 可以首先设置一个较小的簇数量，这样可以分析每个簇

# 用k均值提取簇
km = KMeans(n_clusters=10, random_state=0)
label_km = km.fit_predict(X_pca)
print('Cluster sizes k-means:{}'.format(np.bincount(label_km)))
# 将簇中心可视化来分析k均值的结果
fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(12, 4))
for center, ax, in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),
              vmin=0, vmax=1)
plt.show()

# 对每个簇中心给出5张最典型图像（簇中心中与簇中心最近的点）
# 与5张最不典型的图像
mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
plt.show()

# 用凝聚聚类分析人脸数据集
# 用ward凝聚聚类提取簇
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print('Cluster sizes agglomerative clustering:{}'.format(
    np.bincount(labels_agg)
))
#

# 通过计算ARI来度量凝聚聚类和k均值给出的两种数据划分是否相似
print('ARI: {:.2f}'.format(adjusted_rand_score(labels_agg, label_km)))
#

# 将10个簇可视化
n_clusters = 10
for cluster in range(n_clusters):
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(15, 8))
    axes[0].set_ylabel(np.sum(mask))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                      labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={'fontsize': 9})
plt.show()

# 使用40个簇，挑出一些特别有趣的簇
# 使用ward凝聚聚类提取簇
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_aff = agglomerative.fit_predict(X_pca)
print('cluster sizes agglomerative clustering:{}'.format(np.bincount(labels_agg)))
n_clusters = 40
for cluster in [10, 13, 19, 22, 36]:  # 手动挑选有趣的簇
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel('#{}:{}'.format(cluster, cluster_size))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                      labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape))
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={'fontsize': 9})
    for i in range(cluster_size, 15):
        axes[i].set_visible(False)
plt.show()




