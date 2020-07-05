# 【导入相应的库（对数据库进行切分需要用到的库是sklearn.model_selection 中的 train_test_split）】
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------【若标签为Striing,先将标签转化为整型】------------------------------


def iris_type(s):
    class_label = {b'Iris-setosa': 0,
                   b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return class_label[s]


# 使用numpy中的loadtxt读入数据文件（csv格式的iris数据，也可直接换成txt格式）
filepath = 'iris.data'  # 数据文件路径
data = np.loadtxt(filepath, dtype=float, delimiter=',',
                  converters={4: iris_type})
# -------------------------------------------------------------------------------------

# 【将矩阵最后一列之前的数值给X（输入数据），将矩阵最后一列的数值给y（标签）】
X, y = data[:, :-1], data[:, -1]

# 【利用train_test_split方法，将X,y随机划分为训练集（X_train），训练集标签（y_train），测试集（X_test），测试集标签（y_test），按训练集：测试集=7:3的概率划分，到此步骤，可以直接对数据进行处理】
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 【将训练集与数据集的数据分别保存为CSV文件】
# np.column_stack将两个矩阵进行组合连接，numpy.savetxt 将txt文件保存为csv格式的文件
train = np.column_stack((X_train, y_train))
np.savetxt('train.csv', train, delimiter=',',
           fmt="%.1f %.1f %.1f %.1f %d")

test = np.column_stack((X_test, y_test))
np.savetxt('test.csv', train, delimiter=',', fmt="%.1f %.1f %.1f %.1f %d")
