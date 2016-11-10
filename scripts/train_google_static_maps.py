import glob
import numpy as np
import sys
from PIL import Image
from skimage.measure import block_reduce
from scipy import misc
from typing import List

from train import create_args, get_model_optimizer

from chainer import training, serializers
from chainer.training import extensions


import argparse
import imp
import logging
import os
import re
import shutil
import time
from multiprocessing import Process
from multiprocessing import Queue

import chainer
import six
from chainer import Variable
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import lmdb
from draw_loss import draw_loss
from utils.transformer import transform


def grid_split(arr, num_channel, debug=False):
    """Split 640x640x3 image into 100 64x64x3 image patches"""
    chunk = np.array(np.split(arr, 10, axis=0))
    chunk = np.array(np.split(chunk, 10, axis=2))

    if debug:

        img = Image.fromarray(chunk[0, 0, :, :, :])
        img.show()

    if num_channel == 3:
        return np.reshape(chunk, (100, 64, 64, 3))
    else:
        return np.reshape(chunk, (100, 64, 64))


def prepare_image_patches(dataset_dir: str='../data/google_static_maps/', training:bool=True) -> (List, List):
    """Prepare 64x64 patch images as numpy arrays. This method assumes the image size is 640x640"""
    if training:
        training_dir = dataset_dir + ''
    else:
        training_dir = dataset_dir + 'test/'
    image_filenames = glob.glob(training_dir + 'images/*.png')

    images = []
    masks = []

    # Read image files and mask files
    for im_filename in image_filenames:
        mask_filename = im_filename.replace('images', 'masks')
        im = misc.imread(im_filename).astype(np.float32)
        images.append(im)

        ma = misc.imread(mask_filename).astype(np.int32)
        masks.append(ma)

    # Normalize images
    images = np.array(images)
    images = (images - images.mean()) / images.std()

    # Split the images and masks into patches
    image_patches = []
    mask_patches = []
    for im, ma in zip(images, masks):
        im_patches = grid_split(im, 3)
        image_patches.append(im_patches)

        ma_patches = grid_split(ma, 1)
        mask_patches.append(ma_patches)

    image_patches = np.vstack(image_patches)
    mask_patches = np.vstack(mask_patches)
    return image_patches, mask_patches


def load_training_data():
    """Load data"""
    dataset_dir = '../data/google_static_maps/'
    image_patches, mask_patches = prepare_image_patches(dataset_dir, training=True)

    # Swap axes of image patches to conform with CNN model input. Downsample the label patches to conform with the CNN.
    image_patches = np.swapaxes(np.swapaxes(image_patches, 2, 3), 1, 2)
    shrunk_mask_patches = np.array(list(map(lambda patch: block_reduce(patch, (4, 4), func=np.max), mask_patches)))

    return image_patches, shrunk_mask_patches

def load_testing_data():
    """Load data"""
    dataset_dir = '../data/google_static_maps/'
    image_patches, mask_patches = prepare_image_patches(dataset_dir, training=False)

    # Swap axes of image patches to conform with CNN model input. Downsample the label patches to conform with the CNN.
    image_patches = np.swapaxes(np.swapaxes(image_patches, 2, 3), 1, 2)
    shrunk_mask_patches = np.array(list(map(lambda patch: block_reduce(patch, (4, 4), func=np.max), mask_patches)))

    return image_patches, shrunk_mask_patches

def create_mini_batch_queue(image_patches, mask_patches):
    """Create a set of mini batches to feed into the training algorithm"""
    dataset_size = image_patches.shape[0]
    batch_count = dataset_size // 100
    index_array = np.arange(dataset_size)
    np.random.shuffle(index_array)
    batch_indices = index_array.reshape(batch_count, 100)

    batch_queue = []
    for idx in range(batch_count):
        batch = (
            image_patches[batch_indices[idx], :, :, :],
            mask_patches[batch_indices[idx],:,:]
        )
        batch_queue.append(batch)

    return batch_queue

def create_mini_batch_iterator(image_patches, mask_patches, batch_size=100, repeat=True, shuffle=True):
    from chainer.datasets import tuple_dataset
    data = tuple_dataset.TupleDataset(image_patches, mask_patches)
    return chainer.iterators.SerialIterator(data, batch_size=batch_size, repeat=repeat, shuffle=shuffle)

def one_epoch(args, model, optimizer, epoch, minibatch_queue, train):
    print("Epoch:%d" % epoch)
    model.train = train
    args.gpu = -1
    xp = cuda.cupy if args.gpu >= 0 else np

    n_iter = 0
    sum_loss = 0
    num = 0
    while True:

        if len(minibatch_queue) == 0:
            break

        x, t = minibatch_queue.pop()

        volatile = 'off' if train else 'on'
        x = Variable(xp.asarray(x), volatile=volatile)
        t = Variable(xp.asarray(t), volatile=volatile)

        if train:
            optimizer.update(model, x, t)
        else:
            model(x, t)

        sum_loss += float(model.loss.data) * t.data.shape[0]
        num += t.data.shape[0]
        n_iter += 1

        del x, t

    # if train and (epoch == 1 or epoch % args.snapshot == 0):
    #     model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
    #     opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
    #     serializers.save_hdf5(model_fn, model)
    #     serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info(
            'epoch:{}\ttrain loss:{}'.format(epoch, sum_loss / num))
    else:
        logging.info(
            'epoch:{}\tvalidate loss:{}'.format(epoch, sum_loss / num))

    return model, optimizer

def main():
    args = create_args()
    args.model = '../models/MnihCNN_multi.py'
    args.epoch = 5
    args.out = 'result'
    args.gpu = -1

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()


    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(args.seed)
    np.random.seed(args.seed)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)

    # prepare dataset
    tr_image_patches, tr_shrunk_mask_patches = load_training_data()
    # te_image_patches, te_shrunk_mask_patches = load_testing_data()
    train_iter = create_mini_batch_iterator(tr_image_patches, tr_shrunk_mask_patches)

    # Ok, I'm using trainging dataset for testing just for debugging.
    test_iter = create_mini_batch_iterator(tr_image_patches, tr_shrunk_mask_patches, repeat=False, shuffle=False)

    # set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    # for epoch in six.moves.range(num_epoch):
    #     queue = create_mini_batch_queue(image_patches, shrunk_mask_patches)
    #     model, optimizer = one_epoch(args, model, optimizer, epoch, queue, True)
    xp = cuda.cupy
    te_image_patches = xp.asarray(te_image_patches)

    te_image_patches = te_image_patches[:100, :, :, :]
    x = chainer.Variable(te_image_patches)
    model.train = False
    model(x, None)
    return


def test():
    args = create_args()
    args.model = '../models/MnihCNN_multi.py'
    args.epoch = 5
    args.out = 'result'
    args.gpu = 0

    cuda.get_device(0).use()

    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(args.seed)
    np.random.seed(args.seed)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)
    model.train = False

    serializers.load_npz('npz/test.npz', model)
    te_image_patches, te_shrunk_mask_patches = load_testing_data()
    te_image_patches = xp.asarray(te_image_patches)
    te_shrunk_mask_patches = xp.asarray(te_shrunk_mask_patches)

    te_image_patches = te_image_patches[:100,:,:,:]
    x = chainer.Variable(te_image_patches)
    model(x, None)
    return

if __name__ == '__main__':
    main()
