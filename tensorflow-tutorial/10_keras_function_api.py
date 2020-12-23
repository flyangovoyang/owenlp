import numpy as np
import tensorflow as tf
from tensorflow import keras


# 函数式api支持非线性拓扑的模型，支持共享层，支持多输入和多输出。
# 首先需要定义输入层，专门用来接收数据
inputs = keras.Input(shape=(784,))
dense = keras.layers.Dense(64, activation='relu')
x = dense(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

keras.utils.plot_model(model, 'model.png', show_shapes=True)