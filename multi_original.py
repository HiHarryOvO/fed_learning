"""
This file is based on the fed_learning.py file which implements a federated learning system with certain number of
clients.

Work is based on AN Bhagoji et al's Analyzing Federated Learning through
an Adversarial Lens published on PMLR. We are trying to discover more research opportunities on this federated
learning attack topic.

The implementation of federated learning is done by myself and sorry for not organizing this whole algorithm into
several files to make it more comfortable to read and manage. Might work on this later.

Dataset used here is MNIST dataset.
"""

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

# %matplotlib inline

sns.set()

num_clients = 10
mal_num = 6

"""
An instance of this class Server doesn't have to be operating as a server. On the contrary, an instance as class Server 
might only indicate that this instance also as an Sequential object has the same structure as model in the server and 
doesn't work as a client for some reason. Therefore we don't number it or weight it.
"""


class BaseModel(Sequential):
    def __init__(self, loss='categorical_crossentropy'):
        super().__init__()

        # Adding the CNN structures for the model
        self.add(Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(32, (5, 5), activation="relu"))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(10, activation='softmax'))

        # Loss function part
        self.loss = loss
        self.sgd_temp = SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False)
        self.compile(loss=self.loss, optimizer=self.sgd_temp, metrics=["accuracy"])


class Client(BaseModel):
    def __init__(self):
        super().__init__()

        self.weight = 1 / num_clients

        global client_num  # Number all the clients
        self.num = client_num
        self.xdata = x_datasets[self.num]
        self.ydata = y_datasets[self.num]
        client_num += 1

        self.mal_model = BaseModel()
        self.x_aux = None
        self.y_aux = None
        self.m_update = np.array(self.mal_model.get_weights())
        self.mean_benign = 0

        self.mal_epochs = 10
        self.rho = 1e-2

    def train(self):
        self.compile(loss=self.adverse_custom_loss(), optimizer=self.sgd_temp, metrics=["accuracy"])
        self.fit(self.xdata, self.ydata,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=0,
                 validation_data=(x_test, y_test))

        self.mal_model.compile(loss=self.adverse_custom_loss(), optimizer=self.sgd_temp, metrics=["accuracy"])
        self.mal_model.fit(self.x_aux, self.y_aux,
                           batch_size=batch_size,
                           epochs=self.mal_epochs,
                           verbose=0,
                           validation_data=(self.x_aux, self.y_aux))
        self.m_update = np.array(self.mal_model.get_weights()) - w_server

    def adverse_custom_loss(self):
        def loss(y_true, y_pred):
            loss_1 = num_clients * tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

            y_true_1 = tf.convert_to_tensor(self.ydata, np.float32)

            y_pred_1 = self.mal_model.predict(x=self.xdata)
            y_pred_1 = tf.convert_to_tensor(y_pred_1, np.float32)

            loss_2 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_1, y_pred_1))

            upd = np.array(self.mal_model.get_weights()) - w_server
            loss_3 = self.rho * np.linalg.norm(flat(upd - self.mean_benign), ord=2)

            # print("delta:", np.linalg.norm(flat(upd), ord=2))

            return loss_1 + loss_2 + loss_3

        # Return a function
        return loss

    def stealthy_loss(self):
        def loss(y_true, y_pred):
            loss_1 = num_clients * tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

            y_true_1 = tf.convert_to_tensor(self.ydata, np.float32)

            y_pred_1 = self.mal_model.predict(x=self.xdata)
            y_pred_1 = tf.convert_to_tensor(y_pred_1, np.float32)

            loss_2 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true_1, y_pred_1))

            upd = np.array(self.mal_model.get_weights()) - w_server
            loss_3 = self.rho * np.linalg.norm(flat(upd - self.mean_benign), ord=2)

            # print("delta:", np.linalg.norm(flat(upd), ord=2))

            return loss_1 + loss_2 + loss_3

        # Return a function
        return loss


# this function is for one user per epoch and returns the weights or updated delta of weights
def federated_learning(client_online, epoch):
    if epoch == 1:
        client_online.train()
        return np.array(client_online.mal_model.get_weights())
    else:
        client_online.mal_model.set_weights(w_server)
        client_online.train()

        # return delta_w
        return client_online.m_update


# transform model parameters to one dimension to calculate l2-norm
def flat(nums):
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


start = time.clock()  # calculate running time

batch_size = 128
num_classes = 10
epochs = 2

loss1_list = []
loss2_list = []
num_aux = 10
# list_num_aux = [50, 100]
training_rounds = 10

# data processing
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

test_acc = []
aux_acc = []

print("****************************")
print("****************************")
print("AUXILIARY DATASET SIZE:", num_aux)
print("****************************")
print("****************************")

# create auxiliary data
aux_indices = np.random.choice(len(x_test), num_aux * num_clients)
x_aux = x_test[aux_indices]
y_aux_true_uncat = y_test_uncat[aux_indices]
y_aux_false_uncat = np.array([])

"""
Create all the models we need, including clients and central model.
Notice that when compiling mal_model of every client, parameters need to be passed to custom_loss().
"""
# create a dictionary to store all clients
dict_clients = {}
client_num = 0  # starting from 0 to (num_clients - 1)
for i in range(num_clients):
    ind = 'client_' + str(i)
    dict_clients[ind] = Client()

    order = list(range(10))
    random.shuffle(order)

    for j in range(i * num_aux, (i + 1) * num_aux):
        y_aux_false_uncat = np.append(y_aux_false_uncat, order[y_aux_true_uncat[j]])
    dict_clients[ind].x_aux = x_aux[range(i * num_aux, (i + 1) * num_aux)]
    dict_clients[ind].y_aux = keras.utils.to_categorical(y_aux_false_uncat[range(i * num_aux, (i + 1) * num_aux)],
                                                         num_classes)
y_aux_false = keras.utils.to_categorical(y_aux_false_uncat, num_classes)

# build central model
model_server = BaseModel()

for i in range(1, training_rounds + 1):
    print("辅助集 %d 条： 第 %d 次联邦学习" % (num_aux, i))
    if i == 1:
        list_w = []
        w_server = 0
        upd_epoch1 = 0
        for key, client in dict_clients.items():
            w_client = federated_learning(client, epoch=i)
            upd_epoch1 += client.weight * w_client
        w_server = upd_epoch1

        for key, client in dict_clients.items():
            client.mean_benign = (w_server - client.weight * client.m_update) / (num_clients - 1)
        model_server.set_weights(w_server)

        score_test = model_server.evaluate(x_test, y_test, verbose=0, batch_size=32)
        print('model Test loss:', score_test[0])
        print('model Test accuracy:', score_test[1])
        test_acc.append(score_test[1])

        score_aux = model_server.evaluate(x_aux, y_aux_false, verbose=0, batch_size=32)
        print('Aux accuracy:', score_aux[1])
        aux_acc.append(score_aux[1])
    else:
        w_server = np.array(model_server.get_weights())
        delta_w_server = 0
        for key, client in dict_clients.items():
            delta_w_client = federated_learning(client, epoch=i)
            delta_w_server += client.weight * delta_w_client

        w_server += delta_w_server

        for key, client in dict_clients.items():
            client.mean_benign = (w_server - client.weight * client.m_update) / (num_clients - 1)
        model_server.set_weights(w_server)

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

end = time.clock()
print("running time is %g s" % (end - start))
