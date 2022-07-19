import collections
import os
import h5py
import scipy.io as sio
import tensorflow as tf
import numpy as np
import random
import cv2
from tqdm import tqdm

from scipy.spatial import distance

class SiameseDataSet(object):
    def __init__(self,
                 base_dir,
                 test = False
                 ):

        self.test = test
        self.n  = 128
        self._base_dir = base_dir

        self.PATCH_SIZE = 32
        # the loaded patches
        self._data = dict()

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_train_blocks(self):
        return self._num_train_blocks

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index(self):
        return self._index

    def _get_data(self):
        return self._data

    def _get_blocks(self):
        return self._get_data()['blocks']

    def _get_blocks_transformed(self):
        return self._get_data()['blocks_transformed']

    def _get_matrix(self):
        return self._get_data()['labels']


    def load_by_name(self, name, blocks_size=33, num_channels=1, debug=True):
        
        blocks, blocks_transformed, transform_matrix = self._load_blocks(self._base_dir, name, blocks_size, num_channels)
        # load the labels
        self._data['blocks'] = blocks
        self._data['blocks_transformed'] = blocks_transformed
        self._data['labels']  =  transform_matrix
        
        if debug:
            print('-- Dataset loaded:    %s' % name)
            print('-- Number of patches: %s' % len(self._data['blocks']))
            print('-- Number of labels:  %s' % len(self._data['labels']))

    def _load_blocks(self, base_dir, name, patch_size, num_channels):
        data = h5py.File(base_dir + name + ".mat")
        save_data =data["im"]
        blocks_all = np.transpose(save_data, (3, 1,0,2))/255.
        save_data = data["warped_im"]
        blocks_transformed = np.transpose(save_data, (3,1,0,2))/255.
        save_data = data["transform_matrix"]
        transform_matrix = np.transpose(save_data, (1,0))
        self._num_train_blocks = transform_matrix.shape[0]

        return np.asarray(blocks_all), np.asarray(blocks_transformed), np.asarray(transform_matrix)

    def _create_indices(self, labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in xrange(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def _compute_mean_and_std(self, patches):
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
        # compute the standard deviation
        std = np.std(patches)
        return mean, std

    def normalize_data(self, mean, std):
        #pbar = tqdm(self._data['patches'])
        pbar = tqdm(range(self._data['blocks'].shape[0]))
        for i in pbar:
            pbar.set_description('Normalizing data')
            self._data['blocks'][i] = (self._data['blocks'][i] - mean) / std
            self._data['blocks_transformed'][i] = (self._data['blocks_transformed'][i] - mean) / std

    def generate_stats(self):
        #print('-- Computing dataset mean: %s ...' % name)
        # compute the mean and std of all patches
        blocks = self._get_blocks()
        mean, std = self._compute_mean_and_std(blocks)
        print('-- Computing dataset mean:  ... OK')
        print('-- Mean: %s' % mean)
        print('-- Std : %s' % std)
        return mean, std

    def prune(self, min=2):
        labels = self._get_labels()
        ids, labels = self._prune(labels, min)
        return ids, labels

    def _prune(self, labels, min):
        # count the number of labels
        c = collections.Counter(labels)
        # create a list with globals indices
        ids = range(len(labels))
        # remove ocurrences
        ids, labels = self._rename_and_prune(labels, ids, c, min)
        return np.asarray(ids), np.asarray(labels)

    def _rename_and_prune(self, labels, ids, c, min):
        count, x = 0, 0
        labels_new, ids_new = [[] for _ in range(2)]
        while x < len(labels):
            num = c[labels[x]]
            if num >= min:
                for i in xrange(num):
                    labels_new.append(count)
                    ids_new.append(ids[x + i])
                count += 1
            x += num
        return ids_new, labels_new

    def generate_index(self):
        # retrieve loaded patches and labels
        labels = self._get_matrix()

        index = np.arange(labels.shape[0])
        np.random.shuffle(index)

        self._index = index

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            self._epochs_completed += 1
        # Go to the next epoch
        if start + batch_size > self._num_train_blocks:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            #temp_index = np.arange(self._num_train_patch)
            np.random.shuffle(self._index)
            #self._index = temp_index

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        
        return self._index[start:end]

    def iter_select(self,index):
        # retrieve loaded patches and labels
        index = np.asarray(index)
        np.random.shuffle(index)
        self._num_train_patch = index.shape[0]
        self._index = index
