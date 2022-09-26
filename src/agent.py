from model import YangLeGeYangModel
import tensorflow as tf
import numpy as np
import random

class YangLeGeYangeAgent():
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 0.01
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.build()

    def build(self):
        self.model = YangLeGeYangModel()
        self.loss = tf.reduce_mean((self.model.value - self.target) ** 2)
        self.train_q = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimizer(self.loss)
        self.model.sess.run(tf.lobal_variables_initializer())

    def sample(self, state, feature):
        if np.random.rand() <= self.epsilon:
            action = random.randint(0, len(state['Bright_Block'])-1)
        else:
            act_values = self.model.forward(feature['Bright_Block'], feature['Bright_Block_legal'], feature['Dark_Block'], feature['Dark_Block_legal'], feature['Queue_Block'], feature['Queue_Block_legal'])
            action = np.argmax(act_values[0])
        return action

    def learn(self, training_data):
        feature = training_data['feature']
        self.model.sess.run(self.train_q, feed_dict={self.model.Bright_Block: feature['Bright_Block'], 
                                                        self.model.Bright_Block_legal: feature['Bright_Block_legal'],
                                                         self.model.Dark_Block: feature['Dark_Block'],
                                                         self.model.Dark_Block_legal: feature['Dark_Block_legal'],
                                                         self.model.Queue_Block: feature['Queue_Block'],
                                                         self.model.Queue_Block_legal: feature['Queue_Block_legal'],
                                                         self.target: training_data['target']
                                                         })

