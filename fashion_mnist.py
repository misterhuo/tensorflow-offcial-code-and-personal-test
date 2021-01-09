import tensorflow as  tf #2.1.0
from tensorflow import keras
import numpy as np
import  matplotlib.pyplot as plt

# 该数据集图像是 28x28 的 NumPy 数组，像素值介于 0 到 255 之间。标签是整数数组，介于 0 到 9 之间
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#可以简单的看一下数据的特点
print(train_images.shape)
print(train_labels.shape)
#绘制图像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.xlabel(train_labels[0])
plt.grid(False)
plt.show()

'''数据预处理：请将这些值除以 255。请务必以相同的方式对训练集和测试集进行预处理'''
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 显示训练集中前25个图像
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''构建模型;'''
#该网络的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）。
#将该层视为图像中未堆叠的像素行并将其排列起来。该层没有要学习的参数，它只会重新格式化数据。展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。
#它们是密集连接或全连接神经层。
#第一个 Dense 层有 128 个节点（或神经元）。
#第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组。
#每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
'''模型的编译'''
#在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：
#损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
#优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
#指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer='adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10,verbose=0)

'''比较测试集在模型上的准确率'''
test_loss,test_acc= model.evaluate(test_images,test_labels,verbose=0)
print('\n test_accuracy:',test_acc)
# 这时候会发现过拟合问题

'''进行预测'''
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(class_names[np.argmax(predictions[0])])

'''其绘制成图表，看看模型对于全部 10 个类的预测'''
def plot_image(i,predictions_array,true_label,img):
    predictions_array, true_label, img = predictions_array, true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

'''验证预测结果'''
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

'''使用训练好的模型'''
img = test_images[1]  # img.shape = (28, 28)
'''tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中'''
img = (np.expand_dims(img,0)) # img.shape = (1,28, 28)
predictions_single = probability_model.predict(img)
print(np.argmax(predictions_single))
