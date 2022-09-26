import tensorflow as tf


Block_Number = 16

class YangLeGeYangModel():
    def __init__(self):
        self.Bright_Block = tf.placeholder(dtype=tf.float32, shape=(None, 20, Block_Number + 2))
        self.Bright_Block_legal = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.Dark_Block = tf.placeholder(dtype=tf.float32, shape=(None, 20, Block_Number + 2))
        self.Dark_Block_legal = tf.placeholder(dtype=tf.float32, shape=(None, 20))
        self.Queue_Block =  tf.placeholder(dtype=tf.float32, shape=(None, 7, Block_Number))
        self.Queue_Block_legal = tf.placeholder(dtype=tf.float32, shape=(None, 7))
        self.value = None
        self.build()

    def forward(self, Bright_Block, Bright_Block_legal, Dark_Block, Dark_Block_legal, Queue_Block, Queue_Block_legal):
        return self.sess.run(self.values, feed_dict={self.Bright_Block:Bright_Block, 
                                                        self.Bright_Block_legal:Bright_Block_legal,
                                                         self.Dark_Block:Dark_Block,
                                                         self.Dark_Block_legal: Dark_Block_legal,
                                                         self.Queue_Block: Queue_Block,
                                                         self.Queue_Block_legal: Queue_Block_legal
                                                         })
    
    def build(self):
        Bright_Block_Feature = tf.layers.dense(self.Bright_Block, units=32, activation = tf.tanh) # (None, 20 , 32)
        Bright_Blocks_Feature = tf.reduce_sum(Bright_Block_Feature * self.Bright_Block_legal, -1) # (None, 32)
        Dark_Block_Feature = tf.layers.dense(self.Dark_Block, units=32, activation = tf.tanh) # (None, 20 , 32)
        Dark_Blocks_Feature = tf.reduce_sum(Dark_Block_Feature * self.Dark_Block_legal, -1) # (None,32)
        Queue_Block_Feature = tf.layers.dense(self.Queue_Block, units=32, activation = tf.tanh) # (None, 7 , 32)
        Queue_Blocks_Feature = tf.reduce_sum(Queue_Block_Feature * self.Queue_Block_legal, -1) # (None, 32)
        globla_feature = tf.cancat(Bright_Blocks_Feature, Dark_Blocks_Feature, Queue_Blocks_Feature) # (None, 128)
        globla_feature_repeat_temp = tf.repeat(tf.expand_dims(globla_feature, -1), 20, -1) # (None, 128, 20)
        globla_feature_repeat = tf.transpose(globla_feature_repeat_temp, perm = [0, 2, 1]) # (None, 20, 128)
        feature = tf.cancat(Bright_Block_Feature, globla_feature_repeat) # (None, 20, 160)
        self.value = tf.layers.dense(feature, units=1, activation = None) # (None, 20, 1)


    def set_weights(self, weights):
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(scope=self.scope), weights)}

        self.sess.run(self._nodes, feed_dict=feed_dict)

    def get_weights(self):
        return self.sess.run(tf.trainable_variables(self.scope))
    

