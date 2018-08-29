import tensorflow as tf
from numpy.random import RandomState

'''
自定义损失函数
问题描述：在预测商品销量时，预测多了，商家损失的时生产成本，预测少了，损失的是商品利润。
        一般商品成本和商品利润不会严格相等，假设：商品成本1元，利润10元。
        如果神经网络使用均方误差，很可能此模型无法最大化预期利润，为了最大化预期利润，
        需要将损失函数和利润直接联系起来，注意损失函数定义的是损失，所以要将利润最大化，
        定义的损失函数应该刻画成本或者代价。
        f(x, y) = a(x - y)  x > y 正确答案多于预测答案的代价，a = 10
        f(x, y) = b(y - x) x <= y 正确答案少于预测答案的代价，b = 1
        x:正确答案 y:预测答案
'''

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# 回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义一个单层神经网络的前向传播过程，这里就是简单的加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1

# 自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
# 定义反向传播优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)

#设置回归的正确值为2个输入的和加上一个随机量，不同损失函数在能完全预测正确的时候最低，为区别不同损失函数的效果，加入不可预测的噪音，噪音一般为一个均值为0的小量，这里设置为 -0.05 ~ 0.05
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    # 初始化所以变量
    tf.global_variables_initializer().run()
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataSet_size
        end = min(start + batch_size, dataSet_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print(sess.run(w1))

