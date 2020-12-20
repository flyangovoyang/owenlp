import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # C:/Users/owen/.keras/datasets/mnist.npz
x_train, x_test = x_train/255.0, x_test/255.0  # 转成灰度图片

# 利用序列化api快速构建单一输入输出的模型
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

# 编译，主要是配置一些东西
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # metrics必须是列表

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
