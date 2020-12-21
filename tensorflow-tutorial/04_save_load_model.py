import tensorflow as tf
import tensorflow.keras as keras
import os


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义一个简单的序列模型
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# 创建一个基本的模型实例
model = create_model()
model.summary()

checkpoint_path = "model_save/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # 通过回调训练


# 创建一个基本模型实例，
model = create_model()
# 评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 加载权重，重新评估模型
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# # 在文件名中包含 epoch (使用 `str.format`)
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # 创建一个回调，每 5 个 epochs 保存模型的权重
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)


def save_in_other_way(model):
    # 保存权重
    model.save_weights('./checkpoints/my_checkpoint')

    # 创建模型实例
    model = create_model()

    # 恢复权重
    model.load_weights('./checkpoints/my_checkpoint')

    # 评估模型
    loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


"""
只保存权重
model.save_weights(path)
model.load_weights(path)

保存整个模型
save model格式，和tensorflow serving兼容
model.save(path)
tf.keras.models.load_model(path)
保存的目录包含三个：assets save_model.pb variables

hdf5格式
model.save(path.h5)
tf.keras.models.load_model(path.h5)
"""