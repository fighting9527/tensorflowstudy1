import tensorflow as tf
from tensorflow.python.framework import graph_util

'''
保存模型部分信息
'''

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 导出当前计算图的GraphDef部分，只需要一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉，保存计算节点的名称
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将导出的模型存入文件
    with tf.gfile.GFile("model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())