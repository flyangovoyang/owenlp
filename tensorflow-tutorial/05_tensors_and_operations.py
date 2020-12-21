import tensorflow as tf


# tf2的eager_execution默认开启
print(tf.add(1, 2))
print(tf.matmul([[1]], [[2, 3]]))

# tensorflow的张量可以在GPU上获得加速
# 张量是不可修改的
# 张量可以和numpy互转

# 显卡可以加速张量的计算，以下为例
# 检查是否有GPU，之前提到过
# 指定显卡
import time


def time_matmul(x):  # 计算，并记录耗时
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print("10 loops: {:0.2f}ms".format(1000 * result))


# 首先在CPU上运行,需要102ms
with tf.device('CPU:0'):
    x = tf.random.uniform([100, 100])
    assert x.device.endswith('CPU:0')
    time_matmul(x)

# 然后在GPU上运行,需要231ms
if tf.config.experimental.list_physical_devices('GPU'):
    print('on GPU:')
    with tf.device('GPU:0'):
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith('GPU:0')
        time_matmul(x)

# 所以这是个失败的对比例子？？？

# 数据集 Datasets
# 如何创建
# 之前也提到过一个tf.data.Datasets.from_tensor_slices()