import tensorflow as tf
from tensorflow.python.platform import gfile

'''
从模型文件中加载
'''

with tf.Session() as sess:
    model_filename = "model/combined_model.pb"

    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前的图中，返回张量的名称
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))