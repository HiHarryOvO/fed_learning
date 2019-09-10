"""
This file is based on the fed_learning.py file which implements a federated learning system with certain number of
clients. In this file,
we are going to introduce a malicious client who is trying to lead the training of model to mis-classification.

Dataset used here is MNIST dataset.
"""

import os
import logging
import numpy as np
import keras
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K

# logging.getLogger().setLevel(logging.DEBUG)
# logging.basicConfig(filename=os.path.join(os.getcwd(), 'log.txt'), level=logging.DEBUG)

sns.set()

num_clients = 10
mal_num = 6

"""
An instance of this class Server doesn't have to be operating as a server. On the contrary, an instance as class Server 
might only indicate that this instance also as an Sequential object has the same structure as model in the server and 
doesn't work as a client for some reason. Therefore we don't number it or weight it.
"""


class Server(Sequential):
    def __init__(self):
        super().__init__()
        self.add(Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(32, (5, 5), activation="relu"))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(10, activation='softmax'))
        self.sgd_temp = SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False)
        self.compile(loss='categorical_crossentropy', optimizer=self.sgd_temp, metrics=["accuracy"])


class Client(Server):
    def __init__(self):
        super().__init__()

        self.weight = 1 / num_clients

        global client_num
        self.num = client_num
        self.xdata = x_datasets[self.num]
        self.ydata = y_datasets[self.num]
        self.loss1 = []
        client_num += 1

    def train(self):
        self.fit(self.xdata, self.ydata,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=0,
                 validation_data=(x_test, y_test))


# this function is for one user per epoch and returns the weights or updated delta of weights
def federated_learning(client_online, epoch):
    if epoch == 1:
        client_online.train()

        return np.array(client_online.get_weights())
    else:
        w_mod_central = model_server.get_weights()

        client_online.set_weights(w_mod_central)
        client_online.train()

        w_mod_temp = client_online.get_weights()

        # return delta_w
        return np.array(w_mod_temp) - np.array(w_mod_central)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom

    # sim = 0.5 + 0.5 * cos
    return cos


def flat(nums):
    if type(nums) == int:
        return [0 for it in range(len(flat(m_update)))]
    res = []
    for j in nums:
        if isinstance(j, np.ndarray):
            res.extend(flat(j))
        else:
            res.append(j)
    return res


def transform(prediction, dataset):
    result = [[0 for i in range(10)] for j in range(len(dataset))]
    for i in range(len(dataset)):
        pred = prediction[i].tolist()
        result[i][pred.index(max(pred))] = 1
    return result


def krum(list_update):
    scores = []
    for update in list_update:
        distances_from_other_updates = [(flat(update) - flat(other_update)) ** 2 for other_update in list_update if other_update != update]
        sorted_distances = sorted(distances_from_other_updates, reverse=False) # 升序排列
        scores.append(sum(sorted_distances[:num_clients - 3]))
    if scores.index(min(scores)) == len(scores) - 1:
        mal_chosen_times += 1
    return update[scores.index(min(scores))]


def custom_loss(m_update, mean_benign):
    def loss(y_true, y_pred):

        loss_1 = num_clients * tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

        # loss_1 = K.print_tensor(loss_1, message='loss_1 = ')

        y_true_1 = tf.convert_to_tensor(y_datasets[mal_num], np.float32)

        y_pred_1 = mal_model.predict(x=x_datasets[mal_num])
        y_pred_1 = tf.convert_to_tensor(y_pred_1, np.float32)

        loss_2 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_1, y_pred_1))

        # loss_2 = K.print_tensor(loss_2, message='loss_2 = ')

        # m_update = (np.array(mal_model.get_weights()) - w_server) / num_clients
        if i == 1:
            loss_3 = 0
        else:
            loss_3 = -1 * rho * cos_sim(flat(m_update), flat(mean_benign))
        print("余弦相似度：", -1 * loss_3 / rho)
        # print("loss1:", loss_1)
        # print("loss2:", loss_2)
        # print("delta:", np.linalg.norm(flat(m_update), ord=2))

        return loss_1 + loss_2 + loss_3

    # Return a function
    return loss


start = time.clock()  # 计算开始时间

batch_size = 128
num_classes = 10
epochs = 2
mal_epochs = 10
rho = 1

loss1_list = []
loss2_list = []
num_aux = 10
# list_num_aux = [10, 50, 100]
training_rounds = 50

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train_uncat), (x_test, y_test_uncat) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_uncat, num_classes)
y_test = keras.utils.to_categorical(y_test_uncat, num_classes)

# split the data into num_clients ones
length = len(x_train)
x_datasets = []
y_datasets = []

for i in range(num_clients):
    x_datasets.append(x_train[int(i / num_clients * length): int((i + 1) / num_clients * length)])
    y_datasets.append(y_train[int(i / num_clients * length): int((i + 1) / num_clients * length)])

# for num_aux in list_num_aux:
test_acc = []
aux_acc = []
mal_chosen_times = 0
# l2_delta_w = []

print("****************************")
print("****************************")
print("AUXILIARY DATASET SIZE:", num_aux)
print("****************************")
print("****************************")
# create auxiliary data
aux_indices = np.random.choice(len(x_test), num_aux)
x_aux = x_test[aux_indices]
y_aux_true_uncat = y_test_uncat[aux_indices]

# assign fake labels to user_2
order = list(range(10))
random.shuffle(order)

# create false labels
y_aux_false_uncat = []
for i in range(num_aux):
    # allowed_labels = list(range(10))
    # allowed_labels.remove(y_aux_true_uncat[i])
    y_aux_false_uncat.append(order[y_aux_true_uncat[i]])

y_aux_false = keras.utils.to_categorical(y_aux_false_uncat, num_classes)

# create a dictionary to store all clients
dict_clients = {}
client_num = 0  # starting from 0 to (num_clients - 1)
for i in range(num_clients):
    dict_clients['client_' + str(i)] = Client()

# build central model
model_server = Server()

mal_model = Server()
mean_benign = 0
m_update = np.array(mal_model.get_weights())
mal_model.compile(loss=custom_loss(m_update, mean_benign), optimizer=mal_model.sgd_temp, metrics=["accuracy"])

l2_ben = []
l2_mal = []

shadow_mod = Server()
shadow_mod.compile(loss='categorical_crossentropy', optimizer=shadow_mod.sgd_temp, metrics=["accuracy"])

for i in range(1, training_rounds + 1):
    print("辅助集 %d 条：第 %d 次联邦学习" % (num_aux, i))
    if i == 1:
        list_w = []
        w_server = 0
        for key, client in dict_clients.items():
            if client.num != mal_num:
                w_client = federated_learning(client, epoch=i)
                list_w.append(w_client)
                w_benign += client.weight * w_client

                shadow_mod.set_weights(w_client)
                y_true = tf.convert_to_tensor(y_aux_false, np.float32)
                y_pred = shadow_mod.predict(x=x_aux)
                y_pred = tf.convert_to_tensor(y_pred, np.float32)

                client.loss1.append(num_clients *
                                    tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred)))

        mal_model.fit(x_aux, y_aux_false,
                      batch_size=batch_size,
                      epochs=mal_epochs,
                      verbose=0,
                      validation_data=(x_aux, y_aux_false))

        mean_benign = w_benign / (num_clients - 1)

        mal_model_w = np.array(mal_model.get_weights())

        list_w.append(mal_model_w) # 这里不用乘以权重，因为已经不是fed_avg了
        # m_update = mal_model_w / num_clients

        model_server.set_weights(krum(list_w))

        l2_ben.append(np.linalg.norm(flat(mean_benign), ord=2))
        l2_mal.append(np.linalg.norm(flat(mal_model_w / num_clients), ord=2))
        # l2_delta_w.append(l2_value)
        # print("l2 norm of delta w:", l2_value)

        score_test = model_server.evaluate(x_test, y_test, verbose=0, batch_size=32)
        print('model Test loss:', score_test[0])
        print('model Test accuracy:', score_test[1])
        test_acc.append(score_test[1])

        score_aux = model_server.evaluate(x_aux, y_aux_false, verbose=0, batch_size=32)
        print('Aux accuracy:', score_aux[1])
        aux_acc.append(score_aux[1])

    else:
        list_delta_w = []
        delta_w_server = 0
        for key, client in dict_clients.items():
            if client.num != mal_num:
                delta_w_client = federated_learning(client, epoch=i)
                list_delta_w.append(delta_w_client)
                # delta_w_server += client.weight * delta_w_client

                shadow_mod.set_weights(model_server.get_weights() + delta_w_client)
                y_true = tf.convert_to_tensor(y_aux_false, np.float32)
                y_pred = shadow_mod.predict(x=x_aux)
                y_pred = tf.convert_to_tensor(y_pred, np.float32)

                client.loss1.append(num_clients *
                                    tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred)))

        w_server = np.array(model_server.get_weights())

        m_update = np.array(mal_model.get_weights()) - mean_benign * (num_clients - 1)

        mal_model.compile(loss=custom_loss(m_update, mean_benign), optimizer=mal_model.sgd_temp,
                          metrics=["accuracy"])
        mal_model.set_weights(w_server)
        mal_model.fit(x_aux, y_aux_false,
                      batch_size=batch_size,
                      epochs=mal_epochs,
                      verbose=0,
                      validation_data=(x_aux, y_aux_false))
        mean_benign = delta_w_server / (num_clients - 1)
        mal_model_delta_w = np.array(mal_model.get_weights()) - w_server
        list_delta_w.append(mal_model_delta_w) # 同理，不乘以权重
        model_server.set_weights(w_server + krum(list_delta_w)) # aggregate all updates

        l2_ben.append(np.linalg.norm(flat(mean_benign), ord=2))
        l2_mal.append(np.linalg.norm(flat(mal_model_delta_w / num_clients), ord=2))

        # l2_delta_w.append(l2_value)
        # print("l2 norm of delta w:", l2_value)

        score_test = model_server.evaluate(x_test, y_test, verbose=0, batch_size=32)
        print('model Test loss:', score_test[0])
        print('model Test accuracy:', score_test[1])
        test_acc.append(score_test[1])

        score_aux = model_server.evaluate(x_aux, y_aux_false, verbose=0, batch_size=32)
        print('Aux accuracy:', score_aux[1])
        aux_acc.append(score_aux[1])

        if i % 10 == 0:
            # 现在看训练最终的central_model在test data上和auxiliary data上的表现情况（准确度随训练次数变化）
            # aux_acc相当于召回率
            sns.lineplot(x=list(range(i)), y=test_acc, color='blue', label='test acc')
            sns.lineplot(x=list(range(i)), y=aux_acc, color='red', label='aux acc')
            plt.title("Comparison of accuracy over %d epochs" % i)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend(loc='upper right')

            plt.show()

sns.distplot(l2_ben, kde=False, color='blue', label='l2_ben')
sns.distplot(l2_mal, kde=False, color='red', label='l2_mal')
plt.legend(loc='upper right')
plt.show()

""""
with tf.Session() as sess:

    for key, client in dict_clients.items():
        if client.num != mal_num:
            plot_loss = []
            for loss in client.loss1:
                plot_loss.append(loss.eval())
            sns.lineplot(x=list(range(training_rounds)), y=plot_loss, color='red')
    plt.title("r=%d Loss1 of benign clients" % num_aux)
    plt.show()
"""

end = time.clock()
print("running time is %g s" % (end - start))
