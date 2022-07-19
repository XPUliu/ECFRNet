
from __future__ import print_function

import tensorflow as tf
import gc
from model import block_reader
from model import ECFRNet 
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import pickle
from tqdm import tqdm
from datetime import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", nargs='?', type=float, default = 0.01,
                    help="learning rate")

parser.add_argument("--training", nargs='?', type=str, default = 'ECFRNet_train_block',
                    help="Training dataset name")

parser.add_argument("--test", nargs='?', type=str, default = 'ECFRNet_test_block',
                    help="Training dataset name")

parser.add_argument("--alpha", nargs='?', type=float, default = 0.1,
                    help="alpha")

parser.add_argument("--beta", nargs='?', type=float, default = 0.1,
                    help="beta")

parser.add_argument("--num_epoch", nargs='?', type=int, default = 25,
                    help="Number of epoch")

parser.add_argument("--position_dim", nargs='?', type=int, default =2,
                    help="Number of embedding dimemsion")

parser.add_argument("--batch_size", nargs='?', type=int, default = 128,
                    help="Number of embedding dimemsion")

parser.add_argument("--block_size", nargs='?', type=int, default = 33,
                    help="Number of embedding size")
args = parser.parse_args() 

# Parameters
start_learning_rate = args.learning_rate
num_epoch = args.num_epoch
display_step = 1000
batch_size =args.batch_size
block_size =args.block_size
num_epoch = args.num_epoch
now = datetime.now()
suffix = 'LR{:1.0e}_alpha{:1.1e}_beta{:1.1e}'.format(start_learning_rate,args.alpha,args.beta) + now.strftime("%Y%m%d-%H%M%S")
position_dim = args.position_dim

train = block_reader.SiameseDataSet('./data/train_pair/')
train.load_by_name(args.training, block_size)

test = block_reader.SiameseDataSet('./data/train_pair/')
test.load_by_name(args.test, block_size)

print('Loading training stats:')
try:
    file = open('./data/stats_%s.pkl'%args.training, 'r')
    mean, std = pickle.load(file)
except:
    print('No precompute stats! Calculate and save the stats from training data.')
    mean, std = train.generate_stats()
    pickle.dump([mean,std], open('./data/stats_%s.pkl'%args.training,"wb"));
print('-- Mean: %s' % mean)
print('-- Std:  %s' % std)

#normalize data
train.normalize_data(mean, std)
test.normalize_data(mean, std)

# get patches
blocks_train = train._get_blocks()
blocks_train_t = train._get_blocks_transformed()

blocks_test  = test._get_blocks()
blocks_test_t = test._get_blocks_transformed()
print(np.shape(blocks_test))
#get gt transform
blocks_train_label = train._get_matrix()
blocks_test_label = test._get_matrix()

train.generate_index()
test.generate_index()
# get matches for evaluation
print('Learning Rate: {}'.format(args.learning_rate))
print('Feature_Dim: {}'.format(args.position_dim))
print('Alpha: {}'.format(args.alpha))
print('beta: {}'.format(args.beta))
#set up network
CNNConfig = {
    "block_size": block_size,
    "position_dim" : position_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "beta" : args.beta,
    "train_flag": True
}
cnn_model = ECFRNet.BlockCNN(CNNConfig)

#time decayed learning rate
global_step = tf.Variable(0, trainable=False)
decay_step  = 1000
#learning_rate =start_learning_rate
learning_rate = tf.train.exponential_decay(start_learning_rate,global_step,
                                          decay_step, 0.98, staircase=False)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9).minimize(cnn_model.cost,global_step=global_step)
saver = tf.train.Saver()
training_logs_dir = './log/tensorflow/train_{}/'.format(suffix)
test_logs_dir = './log/tensorflow/test_{}/'.format(suffix)
# Initializing the variables
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

# Launch the graph
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    step = 1
    training_writer = tf.summary.FileWriter(training_logs_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_logs_dir, sess.graph)
    merged = tf.summary.merge_all()

    for i in range(num_epoch):
        t=0
        epoch_loss = 0
        num_batch_in_epoch = train.num_train_blocks//batch_size
        print('Start iteration {}'.format(i))
        for step in tqdm(range(num_batch_in_epoch)):
            #training 
            index = train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={cnn_model.blocks: blocks_train[index],\
                    cnn_model.blocks_t: blocks_train_t[index], \
                    cnn_model.label: blocks_train_label[index]})

            #Show loss        
            if step>0 and step % display_step == 0:
                step = step+1
                fetch = {
                   "o1" : cnn_model.o11_flat,
                   "o2" : cnn_model.o21_flat,
                   "cost": cnn_model.cost,
                   "loss1": cnn_model.loss1,
                   "loss2": cnn_model.loss2,
                }
               
                summary, result = sess.run([merged,fetch], feed_dict={cnn_model.blocks: blocks_train[index], \
                     cnn_model.blocks_t: blocks_train_t[index], \
                     cnn_model.label: blocks_train_label[index]})
                
                
                training_writer.add_summary(summary,tf.train.global_step(sess, global_step))
                epoch_loss =epoch_loss+result["cost"]

                #show test loss
                index = test.next_batch(batch_size)
                fetch = {
                    "cost": cnn_model.cost,
                    "loss1": cnn_model.loss1,
                    "loss2": cnn_model.loss2,
                    "o1" : cnn_model.o11_flat, 
                }
                
                summary, result = sess.run([merged,fetch], feed_dict={cnn_model.blocks: blocks_test[index], \
                     cnn_model.patch_t: blocks_test_t[index], \
                     cnn_model.label: blocks_test_label[index]})

                test_writer.add_summary(summary, tf.train.global_step(sess, global_step))
        #save model
        if i > 0 and (i+1)%5==0:
            #t=result["cost"]
            model_ckpt = './checkpoint/'+args.training+'_iter_'+ str(i+1) +'_model.ckpt'
            saver.save(sess, model_ckpt)

training_writer.close()
test_writer.close()
