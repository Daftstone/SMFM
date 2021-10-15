from time import strftime
from time import localtime
import os
import tensorflow as tf
import numpy as np
import math
import copy
import scipy.sparse as sp
import argparse

flags = tf.flags
FLAGS = flags.FLAGS


def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[2]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[1.,1.]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--gpu', type=str, default='0', help='gpu index')
    parser.add_argument('--use_mix', type=bool, default=False, help='gpu index')
    parser.add_argument('--learning_decay', type=float, default=0.99, help='learning rate decay ratio')
    parser.add_argument('--adv_ratio', type=int, default=2,
                        help='Learning rate.')
    parser.add_argument('--ratio', type=float, default=1., help='learning rate decay ratio')
    args = parser.parse_args()
    return args


def deep_parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='frappe',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[16]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[1.0,1.0]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--gpu', type=str, default='0', help='gpu index')
    parser.add_argument('--use_mix', type=bool, default=False, help='gpu index')
    parser.add_argument('--adv_ratio', type=int, default=10,
                        help='Learning rate.')
    return parser.parse_args()


def pert_vector_product(ys, xs1, xs2, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs1)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs2)[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs2)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(xs2) \
                    for grad_elem in grads_with_none]
    return return_grads


def hessian_vector_product(ys, xs, v, do_not_sum_up=True, scales=1.):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i] / scales, xs[i])[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products / scales, xs)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads
