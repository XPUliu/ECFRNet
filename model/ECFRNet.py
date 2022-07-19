import tensorflow as tf

class BlockCNN:
    def __init__(self, CNNConfig):
        if CNNConfig["train_flag"]:
            self.blocks = tf.placeholder("float32", [None, CNNConfig["block_size"], CNNConfig["block_size"], 3])
            self.blocks_t = tf.placeholder("float32", [None, CNNConfig["block_size"], CNNConfig["block_size"], 3])
            self.label = tf.placeholder("float32", [None, CNNConfig["position_dim"]])
        else:
            self.blocks = tf.placeholder("float32", [1, None, None, 3])

        self.alpha = CNNConfig["alpha"]
        self.beta = CNNConfig["beta"]
        self.position_dim = CNNConfig["position_dim"]

        with tf.variable_scope("siamese") as scope:
            self.o11,self.o12,self.o13 = self.model(self.blocks)
            if CNNConfig["train_flag"]:
                scope.reuse_variables()
                self.o21,self.o22,self.o23 = self.model(self.blocks_t)
            
        self.o11_flat = tf.reshape(self.o11, [-1, self.position_dim])
        self.o12_flat = tf.reshape(self.o12, [-1, self.position_dim])
        self.o13_flat = tf.reshape(self.o13, [-1, self.position_dim])

        if CNNConfig["train_flag"]:
            self.o21_flat = tf.reshape(self.o21, [-1, self.position_dim])
            self.o22_flat = tf.reshape(self.o22, [-1, self.position_dim])
            self.o23_flat = tf.reshape(self.o23, [-1, self.position_dim])

            self.cost,self.loss1, self.loss2= self.regression_loss()


    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


    def conv2d_layer(self, name, shape, x):
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);
        conv_val = tf.nn.relu(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias)
        return conv_val

    def conv2d_layer_no_relu(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias
        return conv_val
    def _variable_with_weight_decay(self, name, shape, wd):
        dtype = tf.float32
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        #weight_init = tf.truncated_normal_initializer(stddev=1.0)
        var = tf.get_variable(name=name, dtype = tf.float32, \
                shape=shape, initializer = weight_init)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def model(self, x):
        h_conv1 = self.conv2d_layer('conv1', [6, 6, 3, 32], x)
        h_conv2 = self.conv2d_layer('conv2', [5, 5, 32, 32], h_conv1)
        h_conv3 = self.conv2d_layer('conv3', [5, 5, 32, 64], h_conv2)
        h_conv4 = self.conv2d_layer('conv4', [5, 5, 64, 64], h_conv3)
        h_conv5 = self.conv2d_layer('conv5', [5, 5, 64, 128], h_conv4)
        h_conv6 = self.conv2d_layer('conv6', [5, 5, 128, 128], h_conv5)
        h_conv7 = self.conv2d_layer('conv7', [5, 5, 128, 256], h_conv6)

        output2=self.conv2d_layer_no_relu('out2', [8, 8, 128, self.position_dim], h_conv6)

        output1=self.conv2d_layer_no_relu('out1', [4, 4, 256, self.position_dim], h_conv7)
        h_conv8 = self.conv2d_layer('conv8', [4, 4, 256, 256], h_conv7)
        output3 = self.conv2d_layer_no_relu('fc1',[1, 1, 256, self.position_dim],h_conv8)
        return output1,output2,output3

    def regression_loss(self):
        alpha = tf.constant(self.alpha)
        beta = tf.constant(self.beta)
        with tf.name_scope('all_loss'):
            with tf.name_scope('loss1'):
                inver_loss1 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o21_flat,self.label),2),1))
                covariance_loss1 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o21_flat,tf.add(self.o11_flat,self.label)),2),1))
                loss1 = inver_loss1 + covariance_loss1
            #covariance loss for transformed patches
            with tf.name_scope('loss2'):
                inver_loss2 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o22_flat,self.label),2),1))
                covariance_loss2 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o22_flat,tf.add(self.o12_flat,self.label)),2),1))
                loss2 = inver_loss2 + covariance_loss2
            with tf.name_scope('loss3'):
                inver_loss3 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o23_flat,self.label),2),1))
                covariance_loss3 = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o23_flat,tf.add(self.o13_flat,self.label)),2),1))
                loss3 = inver_loss3 + covariance_loss3
            with tf.name_scope('loss'):
                loss = tf.multiply(alpha,loss1)+tf.multiply(beta,loss2)+tf.multiply((1-beta-alpha),loss3)
        #write summary 
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss1', loss1)
        tf.summary.scalar('loss2', loss2)
        return loss, loss1, loss2
