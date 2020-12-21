import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

"""
120,4,setosa,versicolor,virginica
6.4,2.8,5.6,2.2,2
5.0,2.3,3.3,1.0,1
4.9,2.5,4.5,1.7,2
4.9,3.1,1.5,0.1,0

第一行是表头，表示120个样本，共有四个特征，标签名称有三种
"""

# CSV文件中列的顺序
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# 接下来通过tf.data.Dataset来加载数据
batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_names=label_name,
    num_epochs=1
)

# 查看一下数据
features, labels = next(iter(train_dataset))


# 接下来又是一大堆花里胡哨的操作，不用管
def pack_features_vector(features, labels):
    """将特征打包到一个数组中"""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:5])
# 此时的输出为
"""
tf.Tensor(
[[5.  3.5 1.3 0.3]
 [4.8 3.1 1.6 0.2]
 [6.3 2.7 4.9 1.8]
 [7.4 2.8 6.1 1.9]
 [5.  3.2 1.2 0.2]], shape=(5, 4), dtype=float32)
"""

# 接下来创建模型了，这个不是本小节的重点，可以快速阅过
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # 需要给出输入的形式
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# 计算损失值
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)


# 计算损失值和梯度值
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)  # 相当于loss.backward()


train_loss_results = []
train_accuracy_results = []
num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 相当于optimizer.step()

        epoch_loss_avg(loss_value)
        epoch_accuracy(y, model(x))

    ## 由此可见，损失函数、度量函数、优化器，都是用的类对象，损失函数、度量函数二者的类对象都可以直接调用，从而计算出结果

    train_loss_results.append(epoch_loss_avg.result())  # 取值
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))


# 最后把过程总的损失函数和度量画出来，不过可以用tensorboard
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
