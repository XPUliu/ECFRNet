
from __future__ import print_function

import tensorflow as tf
import gc
from model import block_reader
from model import ECFRNet
import numpy as np
from scipy.spatial import distance
import scipy.io as sio
import glob
import pickle
import cv2
import os
import time
import argparse
from datetime import datetime



parser = argparse.ArgumentParser()

parser.add_argument("--train_name", nargs='?', type=str, default = 'ECFRNet_train_block_iter_15',
                    help="Training dataset name")

parser.add_argument("--stats_name", nargs='?', type=str, default = 'ECFRNet_train_block',
                    help="Training dataset name")

parser.add_argument("--dataset_name", nargs='?', type=str, default = 'test',
                    help="Training dataset name")

parser.add_argument("--save_feature", nargs='?', type=str, default = 'Candidate_corner_prediction',
                    help="Training dataset name")

parser.add_argument("--alpha", nargs='?', type=float, default = 0.1,
                    help="alpha")

parser.add_argument("--beta", nargs='?', type=float, default = 0.1,
                    help="beta")

parser.add_argument("--position_dim", nargs='?', type=int, default = 2,
                    help="Number of embedding dimemsion")
args = parser.parse_args()
train_name = args.train_name
stats_name = args.stats_name
dataset_name = args.dataset_name
save_feature_name = args.save_feature 

# Parameters
batch_size = 128
position_dim = args.position_dim

print('Loading training stats:')
file = open('./data/stats_%s.pkl'%stats_name, 'rb')
mean, std = pickle.load(file)
print(mean)
print(std)

CNNConfig = {
    "position_dim" : position_dim,
    "batch_size" : batch_size,
    "alpha" : args.alpha,
    "beta" : args.beta,
    "train_flag" : False
}

cnn_model = ECFRNet.BlockCNN(CNNConfig)

#dataset information
subsets = []
if dataset_name=='test' :
    subsets = ['lab']
working_dir = './data/'
load_dir = working_dir+'datasets'+  '/' + dataset_name
save_dir = working_dir + save_feature_name + '/' + dataset_name

if not os.path.exists(working_dir + save_feature_name + '/'):
    os.mkdir(working_dir + save_feature_name + '/')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    try:
        saver.restore(sess, "./checkpoint/"+train_name+"_model.ckpt")
        print("Model restored.")
    except:
        print('No model found')
        exit()

    for i,subset in enumerate(subsets) :
        index = 1
        if not os.path.exists(save_dir + '/' + subset) :
            os.mkdir(save_dir + '/' + subset)
        for file in glob.glob(load_dir + '/' + subset + '/*.*') :
            file = os.path.basename(file)
            output_list = []
            if file.endswith(".ppm") or file.endswith(".pgm") or file.endswith(".png") or file.endswith(".JPEG") :
                image_name = load_dir +'/' + subset + '/' + file
                print(image_name)
                save_file = file[0:-4] + '.mat'
                save_name =  save_dir  +'/' + subset + '/' + save_file
                
                #read image
                img= img = cv2.imread(image_name)
                if img.shape[2] == 1 :
                    img = np.repeat(img, 3, axis = 2)
                img= cv2.copyMakeBorder(img,16,16,16,16, cv2.BORDER_REFLECT)
                resized=img/255.
                #predict transformation
                fetch = {
                        "o1": cnn_model.o13
                }
                start1 = time.time()
                resized = np.asarray(resized)
                resized = (resized-mean)/std
                #print(np.shape(resized))
                resized = resized.reshape((1,resized.shape[0],resized.shape[1],3))
                #print(np.shape(resized))

                result = sess.run(fetch, feed_dict={cnn_model.blocks: resized})
                end1 = time.time()
                output_list = result["o1"].reshape((result["o1"].shape[1],result["o1"].shape[2],result["o1"].shape[3]))
                sio.savemat(save_name,{'output_list':output_list})
                end=float(end1 - start1)