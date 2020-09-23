import pandas as pd
import numpy as np
import tensorflow as tf
import math

from typing import Union

class DenseNeuralNetwork:
    def __init__(self, loss_func: str = 'Quadratic', optimizer: str = 'SGD', activation: str = 'sigmoid', epochs: int = 10,
        batch_size: int = 32, learning_rate: float = 0.01, batch_norm: bool = False, verbose: bool = False):
        ''''''
        self.loss_function = loss_func
        self.optimizer = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
    
    def _activation(self, activation: str, y_matrix: tf.Tensor):
        ''''''
        if activation == 'Sigmoid' or activation == 'sigmoid':
            return tf.compat.v1.sigmoid(y_matrix)
        elif activation == 'ReLu' or activation == 'relu':
            return tf.compat.v1.nn.relu(y_matrix)
        elif activation == 'Tanh' or activation == 'tanh':
            return tf.compat.v1.tanh(y_matrix)
        elif activation == 'LeakyReLu':
            return tf.compat.v1.nn.leaky_relu(y_matrix)
        elif activation == 'Softmax' or activation == 'softmax':
            return tf.compat.v1.nn.softmax(y_matrix)
        else:
            print("Invalid Activation function, returning original matrix")
            return y_matrix

    def _batch_normalization(self, y_matrix: tf.Tensor, batch_norm: bool = False, is_training: tf.bool = False):
        ''''''
        if is_training and batch_norm:
            return tf.compat.v1.layers.batch_normalization(y_matrix)
        
        return y_matrix
    
    def _loss_function(self, loss_func: str, y_pred: tf.Tensor, y_true: tf.Tensor):
        ''''''
        if loss_func == 'Quadratic' or loss_func == 'quadratic':
            loss_tensor = tf.reduce_mean(tf.square(y_pred - y_true))
            return loss_tensor
        elif loss_func == 'Cross_entropy' or loss_func == 'cross_entropy':
            num_classes = y_pred.get_shape().as_list()[-1]
            flat_pred = tf.reshape(y_pred, [-1, num_classes])
            flat_labels = tf.reshape(y_true, [-1, num_classes])
            loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_pred))
            return loss_tensor
        else:
            raise NotImplementedError('Loss function ' + loss_func + ' is not supported!')

    def _create_hidden_layers(self):
        ''''''
        # Input Layer
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 20], name='x_input')
        self.y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2], name='y_labels')
        self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name='is_training')

        # Hidden Layer 1
        W1 = tf.Variable(tf.random.normal([256, 128], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random.normal([128]), name='b1')

        # Hidden Layer 1 Calculation
        l1 = tf.add(tf.matmul(self.x, W1), b1)
        l1_batch_norm = self._batch_normalization(l1, batch_norm=self.batch_norm, is_training=self.is_training)
        l1_final = self._activation(self.activation, l1_batch_norm)

        # Hidden Layer 2
        W2 = tf.Variable(tf.random.normal([128, 64], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random.normal([64]), name='b2')

        # Hidden Layer 2 Calculation
        l2 = tf.add(tf.matmul(l1_final, W2), b2)
        l2_batch_norm = self._batch_normalization(l2, batch_norm=self.batch_norm, is_training=is_training)
        l2_final = self._activation(self.activation, l2_batch_norm)

        # Hidden Layer 3
        W3 = tf.Variable(tf.random.normal([64, 16], stddev=0.03), name='W3')
        b3 = tf.Variable(tf.random.normal([16]), name='b3')

        # Hidden Layer 3 Calculation
        l3 = tf.add(tf.matmul(l2_final, W3), b3)
        l3_batch_norm = self._batch_normalization(l3, batch_norm=self.batch_norm, is_training=is_training)
        l3_final = self._activation(self.activation, l3_batch_norm)

        # Hidden Layer 4
        W4 = tf.Variable(tf.random.normal([16, 2], stddev=0.03), name='W4')
        b4 = tf.Variable(tf.random.normal([2]), name='b4')

        # Hidden Layer 3 Calculation
        l4 = tf.add(tf.matmul(l3_final, W4), b4)
        y_out = self._activation('Softmax', l4)

        # Loss Function and Optimization
        self.loss_tensor = self._loss_function(self.loss_function, y_out, self.y)
        correct_preds = tf.equal(tf.argmax(y_out, -1), tf.argmax(self.y, -1))
        self.accuracy_tensor = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        update_weights = self.optimizer.minimize(self.loss_tensor)
        self.update_weights = tf.group([update_weights, update_ops])
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        ''''''
        self._create_hidden_layers()

        # Training
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            total_batch_size = math.ceil(x_train.shape[0] / self.batch_size)

            for epoch in range(1, self.epochs + 1):
                train_loss = train_acc = 0.0
                valid_loss = valid_acc = 0.0

                for i in range(total_batch_size):
                    x_train_batch = x_train[i * self.batch_size:i * self.batch_size + self.batch_size]
                    y_train_batch = y_train[i * self.batch_size:i * self.batch_size + self.batch_size]

                    _, _, loss, acc = sess.run([self.update_weights, self.optimizer, self.loss_tensor, self.accuracy_tensor],
                        feed_dict={self.x:x_train_batch, self.y:y_train_batch, self.is_training:True})

                    train_loss += loss
                    train_acc += acc
                
                train_loss /= total_batch_size
                train_acc /= total_batch_size

                valid_loss, valid_acc = sess.run([self.loss_tensor, self.accuracy_tensor])

                if self.verbose and epoch % 10 == 0:
                    self.display_training_statistics(epoch, train_loss, train_acc, valid_loss, valid_acc)
    
    def display_training_statistics(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        ''''''
        print('Epoch: {0}'.format(epoch + 1))
        print('-------------------')
        print('Train Loss = ', '{:.3f}'.format(train_loss), 'Train Accuracy = ', '{:.3f}'.format(train_acc))
        print('Valid Loss = ', '{:.3f}'.format(valid_loss), 'Valid Accuracy = ', '{:.3f}'.format(valid_acc))
        print('-------------------')
        print()


                    




        

    